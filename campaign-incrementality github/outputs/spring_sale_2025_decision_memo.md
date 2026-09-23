# Decision memo: Spring Sale 2025

_Generated from pipeline run. Activation week 2025-03-31; measurement window 2025-04-07 to 2025-05-05 (5 weeks)._

## Headline

Net incrementality: **+2.4%** on GBV (90% interval +0.9% to +3.8%), worth **$8.8M**. Verdict: material effect.

| Build | Rate | Value |
| --- | ---: | ---: |
| Participating properties | +7.6% | $27.9M |
| Less cannibalisation of competitive set | -4.5% | $-19.1M |
| Total incrementality | +2.4% | $8.8M |

Room nights moved +10.7% against booking value's +7.6%; the gap is the discount reaching travellers.

## Drivers of growth

The GBV lift is carried by conversion rate at +8.5% and length of stay at +1.1%, partly given back through average daily rate at -2.9%.

| Driver | Lift | 90% interval | Share of movement |
| --- | ---: | ---: | ---: |
| Traffic (visits) | +0.9% | +0.0% to +1.8% | 7% |
| Conversion rate | +8.5% | +7.9% to +8.8% | 62% |
| Length of stay | +1.1% | +0.9% to +1.3% | 8% |
| Average daily rate | -2.9% | -3.0% to -2.8% | 23% |

Drivers compose to +7.38% against a directly estimated +7.59% — a residual of +0.20%, which reconciles.

## Money

| Line | USD |
| --- | --- |
| Incremental booking value | $8.79M |
| Incremental contribution | $1.10M |
| Cash-funded discount | $-0.85M |
| Fixed campaign cost | $-0.45M |
| Net value | $-0.21M |

Return on campaign spend: **0.84x** (net $-0.2M). Break-even lift: **2.8%**, against a measured 7.6%. Of the $10.7M discount granted, **$9.9M (93%)** went to demand the model says would have converted anyway.

## What the displacement test is worth

| | Gross of displacement | Net of displacement |
| --- | --- | --- |
| Incremental booking value | $27.9M | $8.8M |
| Effective lift | 7.6% | 2.4% |
| Return on spend | 2.68x | 0.84x |
| Net value | $2.19M | $-0.21M |

**These land on opposite sides of break-even.** Accepting the no-displacement assumption without testing it would have booked this campaign as value-creating. Testing it reverses the decision, which makes the comp-set test the analysis rather than a robustness check.

## Falsification scorecard

| Test | Result | Statistic | Detail |
| --- | --- | --- | --- |
| Rolling-origin backtest | PASS | 0.0199 | 22 origins, 5-week horizon, mean absolute error 1.99% |
| In-time placebo | PASS | 0.0070 | fake activation 2024-09-30 (26 weeks early) returns +0.70% |
| In-space placebo | PASS | 0.0909 | 10 untreated cohorts of 1000 properties; null sd 0.53%; observed +7.59% ranks 1/11 |
| Comp-set displacement | PASS | -0.0294 | untreated comp set moves -2.94% over the same weeks; netted off the headline as displacement |
| Control-pool leakage (underpowered) | PASS | -0.0264 | control pool reads -2.6% against unaffected demand, detectable only to +/-1.5% -- rules out gross contamination, cannot confirm the 1-2% that would matter; design does not use the pool as a control |
| Relationship stability | PASS | 0.0189 | forecast error 1.04x from older to recent origins (1.96% to 2.03%); early vs late pre-period counterfactuals diverge 1.89% |
| Specification sensitivity | FAIL | 0.0348 | 9 validated variants span +8.00% to +11.47% (3.48% wide) |

## Stacking

Stack-enabled properties outperformed stack-disabled participants by **+2.2%** (90% CI +1.6% to +2.8%, p=0.000). Parallel-trends check p=0.00 (FAILS -- treat as descriptive).

## What this does not answer

- Post-window carryover: the measurement stops at the campaign end date, so any repeat-booking effect is excluded and the estimate is a lower bound.
- Participation is not randomised. Properties opted into the campaign, so the estimate is the effect on participants, not the effect of extending the campaign to non-participants.
- The discount-depth ceiling holds volume fixed; a shallower discount would presumably have produced less lift.
