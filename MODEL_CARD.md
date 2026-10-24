# Model card: credit card default risk (LightGBM, v1)

## Purpose

Estimate the probability that a credit card customer defaults on next month's payment, and turn it into an
approve or decline recommendation with reasons. Intended as a **portfolio demonstration** of a credit decisioning
workflow. It is not validated for real lending decisions.

## Data

- UCI "default of credit card clients": 30,000 customers of one Taiwanese issuer, April to September 2005, CC BY 4.0.
- Target: default payment in October 2005 (22.1% positive).
- Split: 60% train (18,000), 20% validation (6,000; used for calibration and threshold selection), 20% test (6,000;
  reported only). Stratified by target, seed 42.
- Cleaning: undocumented education codes (0, 5, 6) and marital code 0 are folded into "other".

## Inputs

Credit limit plus 15 behavioural features computed in SQL from six monthly statements:

- Payment delays: worst, latest, months late (6 and 3 months), months with a bill but no payment.
- Repayment: months paid in full, months with no balance, payments / bills, payment / limit.
- Balance: current, average and peak utilisation, monthly balance trend.

**Excluded by design:** sex, age, marital status and education. Education was an input in the first version, but it
appeared as a decline reason ("Education: university"), which is not defensible. Removing it changed test AUC from
0.7884 to 0.7877.

## Model

- LightGBM (num_leaves 15, learning rate 0.02, early stopping) with **monotonic constraints**. Risk is
  non-decreasing in every delay and missed-payment feature, and non-increasing in months paid in full and in the
  repayment ratio.
- Calibration: Platt scaling. It was chosen over no calibration and over isotonic by 5-fold cross-validated ECE on
  the validation set.
- Baseline: L2 logistic regression on standardised features (signed-log for heavy-tailed ones), with isotonic
  calibration.

## Performance (test set)

| Metric | LightGBM | Logistic regression |
|---|---|---|
| AUC (95% bootstrap CI) | 0.788 (0.773 to 0.803) | 0.771 (0.757 to 0.786) |
| Gini | 0.575 | 0.541 |
| KS | 0.444 | 0.415 |
| Brier | 0.132 | 0.136 |
| ECE (10 quantile bins) | 0.011 | 0.013 |

The AUC difference is +0.017 (paired bootstrap 95% CI 0.010 to 0.024).

## Decision policy

The cost assumptions are illustrative and per applicant, as fractions of the credit limit:

- **Approve, customer repays:** earns 4% (a year of net interest and fees).
- **Approve, customer defaults:** loses 30% (drawn balance × loss given default).
- **Decline:** 0.

These imply a break-even PD of 0.04 / 0.34 = 0.118. The threshold that maximises validation profit is **0.15**. On
test it approves 46.6% of applicants, with an 8.6% default rate among them and NT$9.6M profit. A 0.5 cut-off would
lose NT$4.2M. The threshold should be re-derived whenever the cost assumptions change.

## Explanations

Reasons are TreeSHAP contributions to the log-odds, ranked per applicant, with up to four adverse and four
favourable factors. Contributions sum exactly to the model's log-odds, and this is verified by a test. Reason text
is deterministic. The LLM letter may only rephrase these facts; outputs containing numbers absent from the facts
are replaced with a template.

## Fairness audit (test set)

Adverse impact ratio (AIR) is a group's approval rate divided by the highest group's. The equal-opportunity gap (EO)
compares approval rates among customers who actually repaid. Groups with fewer than 100 people are excluded from the
summary.

| Attribute | Min AIR | Max EO gap | Within-group calibration gap |
|---|---|---|---|
| Sex | 0.92 | 3.1 pts | 1.2 pts |
| Age band (<30, 30-44, 45+) | 0.83 | 8.9 pts | 2.0 pts |
| Marital status | 0.99 | 1.9 pts | 1.5 pts |
| **Education** | **0.68** | **16.9 pts** | 0.8 pts |

The within-group calibration gap is the largest difference, in any group, between mean predicted PD and the observed
default rate.

**Open item: education.** The model never sees education and is calibrated within each education group, so the
approval gap largely mirrors observed default rates (high school 26.5%, graduate school 19.2%). A proxy ablation
removing credit limit (a bank-set value plausibly correlated with income and education) gave:

- Equal-opportunity gap: 16.9 to 10.3 points.
- Min AIR: 0.68 to 0.75.
- AUC: 0.788 to 0.779.
- Approval rate: 46.6% to 34.9%.
- Profit: 13% lower.

Whether to accept that trade-off, or use other mitigations, is a policy decision for business and compliance owners.
It is recorded here rather than silently resolved.

## Monitoring

- PSI per feature and on the score against the training population: under 0.10 is stable, 0.10 to 0.25 moderate,
  over 0.25 significant.
- Test vs train: score PSI 0.001.
- Simulated downturn: score PSI 0.263, with delay and repayment-ratio features flagged.
- In production you would also track realised default rates by score band once outcomes mature, and recalibrate or
  retrain when calibration drifts.

## Limitations

- The data comes from one issuer, one country and one six-month window during a card-debt crisis. It is not
  representative of current portfolios.
- There is no out-of-time validation, because the dataset is a single snapshot.
- The cost model ignores credit-line management, recoveries timing, and customer lifetime value.
- The reason codes explain the model, not causation.
- The fairness audit covers four attributes and observational metrics only.
