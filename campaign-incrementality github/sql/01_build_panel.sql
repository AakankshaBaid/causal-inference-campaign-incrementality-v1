-- =====================================================================
-- Campaign incrementality: building the measurement panel
-- =====================================================================
--
-- This is the contract between the warehouse and the Python package. The
-- estimator consumes exactly one table shaped like `campaign_panel` below;
-- everything upstream of that is warehouse work, everything downstream is
-- modelling. Keeping the seam here is what lets the same estimator run
-- against Trino, Snowflake, Spark SQL or BigQuery with no code change --
-- only this file is dialect-specific.
--
-- Written in ANSI SQL with Trino-style date functions. Object names are
-- generic placeholders; swap in your own warehouse's fact and dimension
-- tables.
--
-- Target grain: one row per (property_id, week).
--
-- Columns the estimator requires:
--   property_id, week, cohort, stack_eligible,
--   gbv, room_nights                     -- outcomes
--   outcome_ly                   -- same property, same ISO week, prior year
--   compset_volume               -- control-series input (see step 5)
--   meta_impressions, destination_queries,
--   market_visitors, market_purchases
-- =====================================================================


-- ---------------------------------------------------------------------
-- 0. Parameters
-- ---------------------------------------------------------------------
-- Keep these in one place and pass them from the orchestrator so that the
-- SQL and the YAML config can never disagree about the campaign window.
--
--   :campaign_name      e.g. 'BFCM 2024 Campaign'
--   :activation_week    Monday of the week the deal went live
--   :pre_weeks          78
--   :post_weeks         8


-- ---------------------------------------------------------------------
-- 1. Participating properties, with dropout screening
-- ---------------------------------------------------------------------
-- A property that switched the deal off in week 3 is not a treated unit for
-- weeks 3-8. Leaving partial participants in the treated cohort attenuates
-- the estimate toward zero, and the attenuation is invisible in the output,
-- so the screen has to happen here rather than being noticed later.
CREATE OR REPLACE VIEW campaign_treated_properties AS
WITH enrolled AS (
    SELECT DISTINCT
        bl.property_id,
        DATE_TRUNC('week', bl.deal_created_date) AS activation_week,
        bl.rate_rule_id
    FROM promo_bulk_load_scope AS bl
    JOIN property_attributes AS pa
      ON pa.property_id = bl.property_id
    WHERE bl.bulk_load_type = :campaign_name
      AND pa.property_category = 'Hotel'
      AND pa.is_test_property = FALSE
      AND pa.structure_category = 'Conventional Lodging'
      AND pa.contract_model IN (
            'Hotel Collect', 'Traveler Preference', 'Collect', 'Collect Agency'
          )
),
live_weeks AS (
    -- Count the weeks each deal was genuinely live across the post window.
    SELECT
        e.property_id,
        e.activation_week,
        COUNT(DISTINCT da.snapshot_date) AS live_snapshots,
        MIN(CASE WHEN da.is_promo_live THEN 1 ELSE 0 END) AS live_throughout
    FROM enrolled AS e
    LEFT JOIN deal_attributes_daily AS da
      ON da.rate_rule_id = e.rate_rule_id
     AND da.snapshot_date >= DATE_ADD('week', 1, e.activation_week)
     AND da.snapshot_date <  DATE_ADD('week', :post_weeks, e.activation_week)
    GROUP BY 1, 2
)
SELECT property_id, activation_week
FROM live_weeks
WHERE live_throughout = 1
  AND live_snapshots = :post_weeks - 1;   -- full post window, no gaps


-- ---------------------------------------------------------------------
-- 2. Competitive set: displaced, NOT a control
-- ---------------------------------------------------------------------
-- These are untreated properties sharing a competitive set with a
-- participant. They are the right population for the displacement test and
-- the *wrong* population for the control series, because a campaign that
-- steals share from them depresses them -- and a control series depressed by
-- the treatment biases the estimate upward. Hence two separate views.
CREATE OR REPLACE VIEW campaign_compset_properties AS
WITH compset_map AS (
    SELECT DISTINCT
        c.property_id,
        CONCAT(CAST(c.market_id AS VARCHAR), '-', CAST(c.compset_id AS VARCHAR)) AS compset_key
    FROM property_compset AS c
    JOIN property_attributes AS pa
      ON pa.property_id = c.property_id
    WHERE c.snapshot_date = (SELECT MAX(snapshot_date) FROM property_compset)
      AND pa.property_category = 'Hotel'
),
treated_keys AS (
    SELECT DISTINCT m.compset_key
    FROM compset_map AS m
    JOIN campaign_treated_properties AS t
      ON t.property_id = m.property_id
)
SELECT DISTINCT m.property_id
FROM compset_map AS m
JOIN treated_keys AS k
  ON k.compset_key = m.compset_key
WHERE m.property_id NOT IN (SELECT property_id FROM campaign_treated_properties);


-- ---------------------------------------------------------------------
-- 3. Distant untreated pool: the control series and the placebo pool
-- ---------------------------------------------------------------------
-- Untreated properties that share no competitive set with any participant,
-- so they are exposed to the same market-wide demand but not to the
-- campaign's displacement. Split downstream into a covariate half and a
-- placebo half (see `split_control_pool` in simulate.py) so the in-space
-- placebo test is not built from units already inside the counterfactual.
CREATE OR REPLACE VIEW campaign_distant_properties AS
SELECT pa.property_id
FROM property_attributes AS pa
WHERE pa.property_category = 'Hotel'
  AND pa.is_test_property = FALSE
  AND pa.property_id NOT IN (SELECT property_id FROM campaign_treated_properties)
  AND pa.property_id NOT IN (SELECT property_id FROM campaign_compset_properties)
  -- Keep the pool comparable on the dimensions that drive booking behaviour,
  -- otherwise the control series tracks a different business.
  AND pa.super_region IN (SELECT DISTINCT super_region
                          FROM property_attributes
                          WHERE property_id IN (SELECT property_id
                                                FROM campaign_treated_properties))
  AND pa.unit_count_band IS NOT NULL;


-- ---------------------------------------------------------------------
-- 4. Weekly production by property
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW campaign_weekly_production AS
SELECT
    pa.property_id,
    DATE_TRUNC('week', t.transaction_date) AS week,
    SUM(t.gross_booking_value_usd)           AS gbv,
    SUM(t.room_night_count)                AS room_nights,
    COUNT(DISTINCT t.booking_item_id)      AS transactions
FROM booking_transactions AS t
JOIN property_attributes AS pa
  ON pa.property_id = t.property_id
WHERE t.transaction_date >= DATE_ADD('week', -(:pre_weeks + 53), :activation_week)
  AND t.transaction_date <  DATE_ADD('week',  :post_weeks,       :activation_week)
  AND t.cancellation_flag = FALSE
GROUP BY 1, 2;


-- ---------------------------------------------------------------------
-- 5. Demand covariates
-- ---------------------------------------------------------------------
-- Market-level, not property-level, so they cannot be contaminated by the
-- campaign itself. Metasearch impressions and destination query volume move
-- with underlying travel intent; they are the signal that separates "our
-- campaign worked" from "November was strong".
CREATE OR REPLACE VIEW campaign_weekly_covariates AS
SELECT
    DATE_TRUNC('week', d.demand_date)                     AS week,
    SUM(d.metasearch_impressions)                         AS meta_impressions,
    SUM(d.destination_query_count)                        AS destination_queries,
    SUM(d.market_visitor_count)                           AS market_visitors,
    SUM(d.market_purchase_count)                          AS market_purchases
FROM market_demand_daily AS d
WHERE d.demand_date >= DATE_ADD('week', -(:pre_weeks + 53), :activation_week)
  AND d.demand_date <  DATE_ADD('week',  :post_weeks,       :activation_week)
GROUP BY 1;


-- ---------------------------------------------------------------------
-- 6. Assemble the panel
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE campaign_panel AS
WITH cohorts AS (
    SELECT property_id, 'treated' AS cohort FROM campaign_treated_properties
    UNION ALL
    SELECT property_id, 'compset' AS cohort FROM campaign_compset_properties
    UNION ALL
    SELECT property_id, 'distant' AS cohort FROM campaign_distant_properties
),
base AS (
    SELECT
        p.property_id,
        p.week,
        c.cohort,
        p.gbv,
        p.room_nights,
        p.transactions,
        -- ISO calendar keys, used for the prior-year join below.
        YEAR_OF_WEEK(p.week) AS iso_year,
        WEEK_OF_YEAR(p.week) AS iso_week
    FROM campaign_weekly_production AS p
    JOIN cohorts AS c
      ON c.property_id = p.property_id
),
with_ly AS (
    -- Same property, same ISO week, previous ISO year.
    --
    -- Joined on (iso_year, iso_week) rather than `week - INTERVAL '52' WEEK`.
    -- For Monday-keyed weeks the two usually agree, because 52 weeks is
    -- exactly 364 days -- but they diverge around 53-week ISO years (2020,
    -- 2026), where a fixed offset slides the comparison by a week. A one-week
    -- slide is harmless in June and material in late December, which is
    -- exactly when promotional campaigns run. ISO week 53 has no prior-year
    -- counterpart and correctly returns NULL rather than silently matching
    -- week 52.
    SELECT
        b.*,
        ly.gbv AS outcome_ly
    FROM base AS b
    LEFT JOIN base AS ly
      ON  ly.property_id = b.property_id
      AND ly.iso_year    = b.iso_year - 1
      AND ly.iso_week    = b.iso_week
),
compset_series AS (
    -- The control series: total weekly volume of the distant untreated pool.
    -- This single column does more work than the entire market-covariate
    -- block, because it moves with the same market factor as the treated
    -- cohort week by week rather than proxying for it.
    SELECT
        week,
        SUM(gbv) AS compset_volume
    FROM base
    WHERE cohort = 'distant'
    GROUP BY 1
)
SELECT
    w.property_id,
    w.week,
    w.cohort,
    COALESCE(s.allows_stacking, FALSE) AS stack_eligible,
    w.gbv,
    w.room_nights,
    w.transactions,
    w.outcome_ly,
    cs.compset_volume,
    cv.meta_impressions,
    cv.destination_queries,
    cv.market_visitors,
    cv.market_purchases
FROM with_ly AS w
LEFT JOIN campaign_weekly_covariates AS cv ON cv.week = w.week
LEFT JOIN compset_series            AS cs ON cs.week = w.week
LEFT JOIN property_stacking_config  AS s  ON s.property_id = w.property_id
WHERE w.week >= DATE_ADD('week', -:pre_weeks, :activation_week);


-- ---------------------------------------------------------------------
-- 7. Pre-flight data checks
-- ---------------------------------------------------------------------
-- Run these before the estimator, not after. Every one of them corresponds
-- to a failure that produces a plausible-looking but wrong number rather
-- than an error.
SELECT
    'panel coverage by cohort'                                   AS check_name,
    cohort,
    COUNT(DISTINCT week)                                   AS weeks,
    COUNT(DISTINCT property_id)                            AS properties,
    SUM(CASE WHEN outcome_ly IS NULL THEN 1 ELSE 0 END)    AS null_outcome_ly,
    SUM(CASE WHEN compset_volume IS NULL THEN 1 ELSE 0 END) AS null_compset,
    SUM(CASE WHEN gbv <= 0 THEN 1 ELSE 0 END)              AS non_positive_nbv
FROM campaign_panel
GROUP BY 1, 2
ORDER BY 2;

-- Property counts must be stable week to week. A cohort that grows mid-window
-- means the enrolment screen leaked, and the level shift will be read as
-- campaign effect.
SELECT
    cohort,
    week,
    COUNT(DISTINCT property_id) AS properties
FROM campaign_panel
GROUP BY 1, 2
ORDER BY 1, 2;
