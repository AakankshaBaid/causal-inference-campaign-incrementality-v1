# Decision memo: Mid Year Sale 2025

_Generated from pipeline run. Activation week 2025-06-30; measurement window 2025-07-07 to 2025-08-04 (5 weeks)._

## Headline

Net incrementality: **+3.2%** on GBV (90% interval +1.5% to +4.9%), worth **$12.9M**. Verdict: material effect.

| Build | Rate | Value |
| --- | ---: | ---: |
| Participating properties | +7.9% | $32.2M |
| Less cannibalisation of competitive set | -4.1% | $-19.3M |
| Total incrementality | +3.2% | $12.9M |

Room nights moved +10.8% against booking value's +7.9%; the gap is the discount reaching travellers.

## Drivers of growth

The GBV lift is carried by conversion rate at +8.5% and length of stay at +1.1%, partly given back through average daily rate at -2.7%.

| Driver | Lift | 90% interval | Share of movement |
| --- | ---: | ---: | ---: |
| Traffic (visits) | +0.9% | -0.1% to +1.9% | 7% |
| Conversion rate | +8.5% | +8.1% to +8.9% | 63% |
| Length of stay | +1.1% | +0.9% to +1.2% | 8% |
| Average daily rate | -2.7% | -2.8% to -2.6% | 21% |

Drivers compose to +7.71% against a directly estimated +7.89% — a residual of +0.19%, which reconciles.

## Money

| Line | USD |
| --- | --- |
| Incremental booking value | $12.95M |
| Incremental contribution | $1.62M |
| Cash-funded discount | $-1.33M |
| Fixed campaign cost | $-0.70M |
| Net value | $-0.41M |

Return on campaign spend: **0.80x** (net $-0.4M). Break-even lift: **3.9%**, against a measured 7.9%. Of the $14.8M discount granted, **$13.5M (91%)** went to demand the model says would have converted anyway.

## What the displacement test is worth

| | Gross of displacement | Net of displacement |
| --- | --- | --- |
| Incremental booking value | $32.2M | $12.9M |
| Effective lift | 7.9% | 3.2% |
| Return on spend | 1.98x | 0.80x |
| Net value | $1.99M | $-0.41M |

**These land on opposite sides of break-even.** Accepting the no-displacement assumption without testing it would have booked this campaign as value-creating. Testing it reverses the decision, which makes the comp-set test the analysis rather than a robustness check.

## Falsification scorecard

| Test | Result | Statistic | Detail |
| --- | --- | --- | --- |
| Rolling-origin backtest | PASS | 0.0194 | 22 origins, 5-week horizon, mean absolute error 1.94% |
| In-time placebo | PASS | 0.0187 | fake activation 2025-04-07 (12 weeks early) returns +1.87% |
| In-space placebo | PASS | 0.0526 | 18 untreated cohorts of 1000 properties; null sd 0.45%; observed +7.89% ranks 1/19 |
| Comp-set displacement | PASS | -0.0267 | untreated comp set moves -2.67% over the same weeks; netted off the headline as displacement |
| Control-pool leakage (underpowered) | PASS | -0.0200 | control pool reads -2.0% against unaffected demand, detectable only to +/-1.4% -- rules out gross contamination, cannot confirm the 1-2% that would matter; design does not use the pool as a control |
| Relationship stability | PASS | 0.0194 | forecast error 1.00x from older to recent origins (1.94% to 1.94%); early vs late pre-period counterfactuals diverge 1.94% |
| Specification sensitivity | PASS | 0.0226 | 9 validated variants span +8.98% to +11.24% (2.26% wide) |

## Stacking

Stack-enabled properties outperformed stack-disabled participants by **+2.2%** (90% CI +1.6% to +2.9%, p=0.000). Parallel-trends check p=0.00 (FAILS -- treat as descriptive).

## What this does not answer

- Post-window carryover: the measurement stops at the campaign end date, so any repeat-booking effect is excluded and the estimate is a lower bound.
- Participation is not randomised. Properties opted into the campaign, so the estimate is the effect on participants, not the effect of extending the campaign to non-participants.
- The discount-depth ceiling holds volume fixed; a shallower discount would presumably have produced less lift.
