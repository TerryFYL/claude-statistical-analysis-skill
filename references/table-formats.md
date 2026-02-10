# APA Table Formatting Standards

## General Rules

- Decimals: 2 for statistics, 3 for p-values
- Leading zero: Omit for values that cannot exceed 1 (r, p, β)
- Significance: *p < .05, **p < .01, ***p < .001
- Notes below table explain abbreviations and symbols

## Table 1: Descriptive Statistics (Continuous)

| Variable | M | SD | Min | Max | Skewness | Kurtosis |
|----------|---:|---:|---:|---:|---:|---:|
| Variable 1 | 3.45 | 1.23 | 1.00 | 7.00 | 0.12 | -0.34 |
| Variable 2 | 4.56 | 0.98 | 1.50 | 6.50 | -0.23 | 0.56 |

*Note.* N = 500. M = mean; SD = standard deviation.

## Table 2: Descriptive Statistics (Categorical)

| Variable | n | % |
|----------|---:|---:|
| Gender | | |
| 　Male | 245 | 49.0 |
| 　Female | 255 | 51.0 |
| Education | | |
| 　High school | 120 | 24.0 |
| 　Bachelor's | 280 | 56.0 |
| 　Master's+ | 100 | 20.0 |

*Note.* N = 500.

## Table 3: Correlation Matrix

| Variable | 1 | 2 | 3 | 4 | 5 |
|----------|---:|---:|---:|---:|---:|
| 1. Variable A | — | | | | |
| 2. Variable B | .45*** | — | | | |
| 3. Variable C | .23** | .34*** | — | | |
| 4. Variable D | -.12 | .56*** | .28** | — | |
| 5. Variable E | .67*** | .18* | -.09 | .41*** | — |

*Note.* N = 500. *p < .05. **p < .01. ***p < .001.

## Table 4: Regression Analysis

| Variable | B | SE | β | t | p |
|----------|---:|---:|---:|---:|---:|
| (Constant) | 2.34 | 0.45 | | 5.20 | <.001 |
| Gender | 0.23 | 0.12 | .11 | 1.92 | .056 |
| Age | -0.02 | 0.01 | -.08 | -2.00 | .046 |
| Predictor X | 0.45 | 0.08 | .34 | 5.63 | <.001 |

*Note.* N = 500. R² = .23. Gender coded: 0 = male, 1 = female.

## Table 5: Hierarchical Regression

| Variable | Step 1 | | Step 2 | |
|----------|---:|---:|---:|---:|
| | β | p | β | p |
| Gender | .12 | .034 | .10 | .065 |
| Age | -.08 | .089 | -.06 | .178 |
| Predictor X | | | .34 | <.001 |
| R² | .05 | | .17 | |
| ΔR² | | | .12*** | |

*Note.* N = 500. ***p < .001.

## Table 6: Moderation Analysis

| Variable | B | SE | t | p | 95% CI |
|----------|---:|---:|---:|---:|---:|
| (Constant) | 3.45 | 0.23 | 15.00 | <.001 | [3.00, 3.90] |
| X | 0.34 | 0.08 | 4.25 | <.001 | [0.18, 0.50] |
| M | 0.28 | 0.09 | 3.11 | .002 | [0.10, 0.46] |
| X × M | 0.15 | 0.05 | 3.00 | .003 | [0.05, 0.25] |

*Note.* N = 500. R² = .28. Variables centered before analysis.

**Simple Slopes:**
| Moderator Level | B | SE | t | p |
|-----------------|---:|---:|---:|---:|
| Low (-1 SD) | 0.19 | 0.10 | 1.90 | .058 |
| High (+1 SD) | 0.49 | 0.09 | 5.44 | <.001 |

## Table 7: Mediation Analysis

| Path | B | SE | 95% CI |
|------|---:|---:|---:|
| Total effect (c) | 0.45 | 0.08 | [0.29, 0.61] |
| Direct effect (c') | 0.28 | 0.09 | [0.10, 0.46] |
| Indirect effect (a×b) | 0.17 | 0.05 | [0.08, 0.28] |

*Note.* N = 500. Bootstrap = 5000. CI = bias-corrected confidence interval.

## p-value Formatting

| Value | Format |
|-------|--------|
| p = .0234 | p = .023 |
| p = .0034 | p = .003 |
| p < .001 | p < .001 |
| p = .050 | p = .050 |
| p = .051 | p = .051 |

## Table 8: Reliability Analysis

| Subscale | Items | M | SD | Cronbach's α | CITC Range |
|----------|------:|---:|---:|---:|---:|
| Subscale A | 5 | 3.45 | 0.89 | .85 | .52-.71 |
| Subscale B | 4 | 3.12 | 1.02 | .78 | .45-.63 |
| Subscale C | 6 | 4.01 | 0.76 | .82 | .48-.68 |
| **Total Scale** | **15** | **3.53** | **0.72** | **.91** | — |

*Note.* N = 500. CITC = corrected item-total correlation.

## Table 9: ROC Analysis / Diagnostic Accuracy

| Index | Value | 95% CI |
|-------|------:|---:|
| AUC | .85 | [.80, .90] |
| Optimal Cutoff | 12.5 | — |
| Sensitivity | .82 | [.76, .87] |
| Specificity | .78 | [.72, .84] |
| PPV | .80 | [.74, .85] |
| NPV | .81 | [.75, .86] |
| LR+ | 3.73 | [2.89, 4.82] |
| LR- | 0.23 | [0.17, 0.31] |

*Note.* N = 500 (cases = 250, controls = 250). Cutoff determined by Youden's index.

## Table 10: Survival Analysis (Cox Regression)

| Variable | HR | 95% CI | p |
|----------|---:|---:|---:|
| Age (per year) | 1.03 | [1.01, 1.05] | .008 |
| Gender (Female vs Male) | 0.72 | [0.55, 0.94] | .016 |
| Stage (III vs I-II) | 2.45 | [1.82, 3.30] | <.001 |
| Treatment (New vs Standard) | 0.58 | [0.43, 0.78] | <.001 |

*Note.* N = 500. HR = hazard ratio. CI = confidence interval. Median follow-up: 24 months.

## Table 11: ICC / Inter-rater Reliability

| Measure | ICC Type | ICC | 95% CI | Interpretation |
|---------|----------|----:|---:|---:|
| Total Score | ICC(3,1) | .89 | [.84, .93] | Good |
| Subscale A | ICC(3,1) | .85 | [.78, .90] | Good |
| Subscale B | ICC(3,1) | .76 | [.68, .83] | Moderate |

*Note.* k = 3 raters. N = 50 subjects. ICC = intraclass correlation coefficient.

## Table 12: Power Analysis / Sample Size

| Analysis | Effect Size | α | Power | Required N |
|----------|------------|---:|---:|---:|
| Independent t-test | d = 0.50 | .05 | .80 | 64/group |
| One-way ANOVA (3 groups) | f = 0.25 | .05 | .80 | 159/group |
| Pearson correlation | r = .30 | .05 | .80 | 85 total |
| Multiple regression (3 IVs) | f² = .15 | .05 | .80 | 77 total |

*Note.* Sample sizes calculated using two-sided tests.

## Common Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| M | Mean |
| SD | Standard deviation |
| SE | Standard error |
| CI | Confidence interval |
| B | Unstandardized coefficient |
| β | Standardized coefficient |
| R² | R-squared |
| ΔR² | R-squared change |
| df | Degrees of freedom |
| n | Sample size (subgroup) |
| N | Total sample size |
| HR | Hazard ratio |
| AUC | Area under the curve |
| PPV | Positive predictive value |
| NPV | Negative predictive value |
| LR+ | Positive likelihood ratio |
| LR- | Negative likelihood ratio |
| ICC | Intraclass correlation coefficient |
| CITC | Corrected item-total correlation |
| α | Cronbach's alpha |
