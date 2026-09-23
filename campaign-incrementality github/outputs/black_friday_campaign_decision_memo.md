# Decision memo: Black Friday Campaign

_Generated from pipeline run. Activation week 2024-11-25; measurement window 2024-12-02 to 2025-01-13 (7 weeks)._

## Headline

Net incrementality: **+5.3%** on GBV (90% interval +3.8% to +6.9%), worth **$19.7M**. Verdict: material effect.

| Build | Rate | Value |
| --- | ---: | ---: |
| Participating properties | +9.0% | $33.3M |
| Less cannibalisation of competitive set | -3.3% | $-13.6M |
| Total incrementality | +5.3% | $19.7M |

Room nights moved +11.9% against booking value's +9.0%; the gap is the discount reaching travellers.

## Drivers of growth

The GBV lift is carried by conversion rate at +8.4% and traffic (visits) at +2.5%, partly given back through average daily rate at -2.5%.

| Driver | Lift | 90% interval | Share of movement |
| --- | ---: | ---: | ---: |
| Traffic (visits) | +2.5% | +1.5% to +3.9% | 18% |
| Conversion rate | +8.4% | +8.1% to +8.9% | 58% |
| Length of stay | +0.9% | +0.8% to +1.0% | 6% |
| Average daily rate | -2.5% | -2.6% to -2.3% | 18% |

Drivers compose to +9.31% against a directly estimated +9.03% — a residual of -0.29%, which reconciles.

## Money

| Line | USD |
| --- | --- |
| Incremental booking value | $19.73M |
| Incremental contribution | $2.47M |
| Cash-funded discount | $-2.31M |
| Fixed campaign cost | $-1.20M |
| Net value | $-1.04M |

Return on campaign spend: **0.70x** (net $-1.0M). Break-even lift: **7.5%**, against a measured 9.0%. Of the $23.1M discount granted, **$20.4M (88%)** went to demand the model says would have converted anyway.

## What the displacement test is worth

| | Gross of displacement | Net of displacement |
| --- | --- | --- |
| Incremental booking value | $33.3M | $19.7M |
| Effective lift | 9.0% | 5.3% |
| Return on spend | 1.19x | 0.70x |
| Net value | $0.66M | $-1.04M |

**These land on opposite sides of break-even.** Accepting the no-displacement assumption without testing it would have booked this campaign as value-creating. Testing it reverses the decision, which makes the comp-set test the analysis rather than a robustness check.

## Falsification scorecard

| Test | Result | Statistic | Detail |
| --- | --- | --- | --- |
| Rolling-origin backtest | PASS | 0.0206 | 20 origins, 7-week horizon, mean absolute error 2.06% |
| In-time placebo | PASS | 0.0076 | fake activation 2024-09-02 (12 weeks early) returns +0.76% |
| In-space placebo | PASS | 0.0909 | 10 untreated cohorts of 1000 properties; null sd 0.58%; observed +9.03% ranks 1/11 |
| Comp-set displacement | PASS | -0.0210 | untreated comp set moves -2.10% over the same weeks; netted off the headline as displacement |
| Control-pool leakage (underpowered) | PASS | -0.0126 | control pool reads -1.3% against unaffected demand, detectable only to +/-1.6% -- rules out gross contamination, cannot confirm the 1-2% that would matter; design does not use the pool as a control |
| Relationship stability | PASS | 0.0300 | forecast error 0.74x from older to recent origins (2.36% to 1.75%); early vs late pre-period counterfactuals diverge 3.00% |
| Specification sensitivity | FAIL | 0.0359 | 9 validated variants span +8.38% to +11.97% (3.59% wide) |

## Stacking

Stack-enabled properties outperformed stack-disabled participants by **+1.7%** (90% CI +1.2% to +2.3%, p=0.000). Parallel-trends check p=0.01 (FAILS -- treat as descriptive).

## What this does not answer

- Post-window carryover: the measurement stops at the campaign end date, so any repeat-booking effect is excluded and the estimate is a lower bound.
- Participation is not randomised. Properties opted into the campaign, so the estimate is the effect on participants, not the effect of extending the campaign to non-participants.
- The discount-depth ceiling holds volume fixed; a shallower discount would presumably have produced less lift.
