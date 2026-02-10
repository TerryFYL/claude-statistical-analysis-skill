# Statistical Methods Index

> This index helps select the appropriate statistical method. Each method includes: applicable scenarios, assumptions, variant selection, and reporting specifications.

---

## Table of Contents

1. [Descriptive Statistics](#1-descriptive-statistics)
2. [Difference Tests](#2-difference-tests)
3. [Correlation Analysis](#3-correlation-analysis)
4. [Regression Analysis](#4-regression-analysis)
5. [Moderation and Mediation](#5-moderation-and-mediation)
6. [Structural Equation Modeling](#6-structural-equation-modeling)
7. [Hierarchical Linear Modeling](#7-hierarchical-linear-modeling)
8. [Meta-Analysis](#8-meta-analysis)
9. [Item Response Theory](#9-item-response-theory)
10. [Longitudinal Analysis](#10-longitudinal-analysis)
11. [Method Selection Decision Tree](#method-selection-decision-tree)

---

## 1. Descriptive Statistics

### 1.1 Central Tendency

| Measure | Applicable Data | When to Use | When to Avoid |
|---------|----------------|-------------|---------------|
| **Mean** | Continuous, normal | Symmetrically distributed data | Extreme values, skewed distribution |
| **Median** | Continuous/ordinal | Skewed distribution, extreme values | When algebraic operations are needed |
| **Mode** | Any type | Categorical variables, bimodal distribution | Usually not reported for continuous variables |
| **Trimmed Mean** | Continuous | Mild extreme values | When extreme values themselves are important |

### 1.2 Dispersion

| Measure | Formula/Description | Applicable Scenario |
|---------|---------------------|---------------------|
| **Standard Deviation (SD)** | sqrt[sum(x-M)^2/(n-1)] | Default choice for normal distribution |
| **Interquartile Range (IQR)** | Q3 - Q1 | Skewed distribution, extreme values |
| **Coefficient of Variation (CV)** | SD/M x 100% | Comparing variability across different units/scales |
| **Range** | Max - Min | Quick overview of data spread |

### 1.3 Distribution Shape

| Measure | Normal Range | Interpretation |
|---------|-------------|----------------|
| **Skewness** | \|S\| < 2 | >0 right-skewed, <0 left-skewed |
| **Kurtosis** | \|K\| < 7 | >0 leptokurtic, <0 platykurtic |

**Normality Tests**:
- N < 50: Shapiro-Wilk
- N >= 50: Kolmogorov-Smirnov
- Large sample (N > 300): Skewness/Kurtosis + Q-Q plot

---

## 2. Difference Tests

### 2.1 Method Selection Matrix

```
                    Number of Groups
                 ┌────┴────┐
              2 Groups    3+ Groups
                 │          │
            ┌────┴────┐    ANOVA Family
       Independent  Paired
            │          │
       ┌────┴────┐  ┌──┴──┐
    Normal    Non-normal  Paired t  Wilcoxon
        │         │
  Independent t  Mann-Whitney
```

### 2.2 Two-Group Comparison

| Method | Assumptions | Alternative When Violated |
|--------|-------------|--------------------------|
| **Independent Samples t-test** | Normality + homogeneity of variance | Welch's t (unequal variance), Mann-Whitney U (non-normal) |
| **Paired Samples t-test** | Normality of differences | Wilcoxon Signed-Rank Test |
| **Welch's t-test** | Normality (does not require equal variance) | Recommended by default, more robust than traditional t |

**Effect Size**:
- Cohen's d = (M1-M2) / SD_pooled
- Interpretation: 0.2 small, 0.5 medium, 0.8 large

### 2.3 Multiple Group Comparison (ANOVA)

| Design Type | Method | Assumptions | Nonparametric Alternative |
|-------------|--------|-------------|--------------------------|
| One-way between-subjects | One-way ANOVA | Normality + equal variance | Kruskal-Wallis |
| One-way within-subjects | Repeated Measures ANOVA | Normality + sphericity | Friedman |
| Two-way between-subjects | Two-way ANOVA | Normality + equal variance | No direct alternative |
| Mixed design | Mixed ANOVA | Normality + sphericity + equal variance | Consider HLM |

**Assumption Tests**:
- Homogeneity of variance: Levene's test (p > .05 satisfied)
- Sphericity: Mauchly's test (p > .05 satisfied)
- Sphericity violated: Greenhouse-Geisser or Huynh-Feldt correction

**Post Hoc Test Selection**:
| Scenario | Recommended Method |
|----------|-------------------|
| Equal variance, equal sample sizes | Tukey HSD |
| Unequal variance | Games-Howell |
| Conservative estimate | Bonferroni |
| Pre-planned comparisons | Planned Contrasts |

---

## 3. Correlation Analysis

### 3.1 Correlation Coefficient Selection

| Variable Combination | Method | Assumptions |
|---------------------|--------|-------------|
| Continuous x Continuous (normal) | **Pearson r** | Linear relationship, bivariate normality |
| Continuous x Continuous (non-normal/ordinal) | **Spearman rho** | Monotonic relationship |
| Ordinal x Ordinal | **Kendall tau** | More robust for small samples |
| Dichotomous x Dichotomous | **Phi** | 2x2 contingency table |
| Dichotomous x Continuous | **Point-biserial** | Equivalent to Pearson |
| Nominal x Nominal | **Cramer's V** | Any contingency table |

### 3.2 Partial and Semi-Partial Correlation

- **Partial Correlation**: The "pure" correlation between X and Y after controlling for Z
- **Semi-Partial (Part) Correlation**: Removing Z's influence only from X

```python
# Partial correlation: X-Y relationship, controlling for Z
partial_corr = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))
```

### 3.3 Effect Size Interpretation

| r Range | Effect Size | Cohen's d Equivalent |
|---------|-------------|---------------------|
| .10 | Small | 0.20 |
| .30 | Medium | 0.50 |
| .50 | Large | 0.80 |

---

## 4. Regression Analysis

### 4.1 Regression Type Selection

| Dependent Variable Type | Method | Link Function |
|------------------------|--------|---------------|
| Continuous (normal) | **Linear Regression (OLS)** | Identity |
| Dichotomous (0/1) | **Logistic Regression** | Logit |
| Multicategorical (unordered) | **Multinomial Logistic** | Logit |
| Multicategorical (ordered) | **Ordinal Logistic** | Cumulative logit |
| Count | **Poisson Regression** | Log |
| Count (overdispersed) | **Negative Binomial Regression** | Log |

### 4.2 Linear Regression Assumption Tests

| Assumption | Test Method | Remedy When Violated |
|------------|-------------|---------------------|
| Linearity | Residual plot, RESET test | Add polynomial terms, variable transformation |
| Normality of residuals | Q-Q plot, Shapiro-Wilk | Can be ignored for large samples, robust SE |
| Homoscedasticity | Breusch-Pagan, residual plot | Robust SE (HC3), WLS |
| No autocorrelation | Durbin-Watson (time series) | GLS, add lagged terms |
| No multicollinearity | VIF < 10, Tolerance > 0.1 | Remove/combine variables, centering |

### 4.3 Model Comparison

- **Nested models**: F test (delta-R-squared), likelihood ratio test
- **Non-nested models**: AIC, BIC (smaller is better)

```
Model Selection Criteria:
- AIC = -2LL + 2k (balances fit and complexity)
- BIC = -2LL + k*ln(n) (penalizes complex models more)
```

### 4.4 Effect Size

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| R-squared | SS_reg / SS_total | Proportion of variance explained |
| Adjusted R-squared | 1 - (1-R²)(n-1)/(n-k-1) | Accounts for number of predictors |
| f-squared | R² / (1-R²) | Cohen's f²: .02 small / .15 medium / .35 large |
| Beta | Standardized coefficient | Allows comparison of relative importance across variables |

---

## 5. Moderation and Mediation

### 5.1 Moderation

**Concept**: W changes the strength or direction of the X to Y relationship

```
        W (Moderator)
        │
        ▼
   X ──────→ Y
```

**Analysis Steps**:
1. Center variables (subtract mean)
2. Create interaction term X x W
3. Regression: Y = b0 + b1*X + b2*W + b3*X*W + epsilon
4. If b3 is significant, moderation effect exists

**Simple Slopes Analysis**:
- Effect of X on Y when W is at M +/- 1SD
- Johnson-Neyman region: Identifies the significance transition point of the moderator

### 5.2 Mediation

**Concept**: M explains how X influences Y

```
        M (Mediator)
       ↗     ↘
   X ──────→ Y
     (Direct Effect)
```

**Modern Method** (recommended): Bootstrap
- Indirect effect = a x b
- 95% CI does not contain 0 implies significance

**Types of Mediation**:
| Type | Condition | Interpretation |
|------|-----------|----------------|
| Full mediation | c' not significant, a*b significant | X influences Y entirely through M |
| Partial mediation | c' significant, a*b significant | X partially influences Y through M |
| No mediation | a*b not significant | M is not a mediator |

### 5.3 Moderated Mediation / Mediated Moderation

**Moderated Mediation**:
- The strength of the mediation effect depends on the moderator
- Conditional indirect effect = a(W) x b or a x b(W)

**Mediated Moderation**:
- The moderation effect is transmitted through a mediator
- Actually a special case of moderated mediation

**Tools**: PROCESS macro (Hayes), lavaan

---

## 6. Structural Equation Modeling (SEM)

### 6.1 Applicable Scenarios

- Multiple dependent variables
- Latent variable measurement
- Complex path relationships
- Theory model validation

### 6.2 Model Types

| Type | Purpose | Characteristics |
|------|---------|-----------------|
| **Path Analysis** | Relationships among observed variables | No latent variables |
| **CFA** | Validate scale structure | Measurement model only |
| **Full SEM** | Comprehensive analysis | Measurement + structural |
| **Multi-group SEM** | Between-group comparison | Measurement invariance |

### 6.3 Fit Indices

| Index | Excellent | Acceptable | Type |
|-------|-----------|------------|------|
| chi-squared/df | < 2 | < 3 | Absolute |
| RMSEA | < .05 | < .08 | Absolute |
| SRMR | < .05 | < .08 | Absolute |
| CFI | > .95 | > .90 | Relative |
| TLI | > .95 | > .90 | Relative |

**Reporting Standard**: Report at least chi-squared(df), RMSEA [90%CI], CFI, SRMR

### 6.4 Common Issues and Solutions

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Non-convergence | Warning messages | Check starting values, simplify model |
| Negative variance | Heywood case | Fix parameters, check data |
| Poor fit | Modification indices | Add paths/correlations guided by theory |
| Insufficient sample size | N < 200 | Simplify model, use MLR |

---

## 7. Hierarchical Linear Modeling (HLM)

### 7.1 Applicable Scenarios

- Nested data (students nested within classrooms)
- Repeated measures (time points nested within individuals)
- Cross-level interactions

### 7.2 Intraclass Correlation Coefficient (ICC)

```
ICC = τ₀₀ / (τ₀₀ + σ²)
```

- ICC > .05 and theoretical nesting structure present implies HLM is recommended
- ICC represents the proportion of between-group variance relative to total variance

### 7.3 Model Building Strategy

1. **Null Model**: Random intercept only, compute ICC
2. **Random Intercept Model**: Add Level-1 predictors
3. **Random Slope Model**: Allow slopes to vary across groups
4. **Cross-level Interaction**: Level-2 variables moderate Level-1 effects

### 7.4 Centering Strategies

| Centering Type | Applicable Scenario | Effect |
|---------------|---------------------|--------|
| **Group-Mean Centering (CWC)** | Level-1 variables | Separates within-group and between-group effects |
| **Grand-Mean Centering (CGM)** | Level-1/2 variables | Preserves original metric |
| **No Centering** | Meaningful zero point | Intercept is interpretable |

---

## 8. Meta-Analysis

### 8.1 Effect Size Conversion

| Original Measure | Convert to d | Convert to r |
|-----------------|--------------|--------------|
| Cohen's d | - | r = d / sqrt(d² + 4) |
| Pearson r | d = 2r / sqrt(1-r²) | - |
| OR | d = ln(OR) * sqrt(3)/pi | r ≈ d / sqrt(d² + 4) |
| t value | d = 2t / sqrt(df) | r = sqrt(t²/(t²+df)) |
| F value (df1=1) | d = 2*sqrt(F/df2) | r = sqrt(F/(F+df2)) |

### 8.2 Model Selection

| Model | Assumption | Applicable Scenario |
|-------|-----------|---------------------|
| **Fixed Effects** | True effect is the same | Homogeneous studies, exploratory |
| **Random Effects** | True effects have a distribution | Recommended by default |
| **Mixed Effects** | Moderators explain heterogeneity | Theoretical predictor variables available |

### 8.3 Heterogeneity Measures

| Measure | Calculation | Interpretation |
|---------|-------------|----------------|
| Q | sum(w*(ES-M)²) | Significant = heterogeneity present |
| I² | (Q-df)/Q x 100% | 25% low / 50% moderate / 75% high |
| tau² | Variance estimate of true effects | Absolute heterogeneity |

### 8.4 Publication Bias Tests

- **Funnel Plot**: Visual inspection of asymmetry
- **Egger's Test**: Regression test (p < .05 indicates bias)
- **Trim-and-Fill**: Estimates missing studies and adjusts

---

## 9. Item Response Theory (IRT)

### 9.1 Model Selection

| Response Format | Number of Parameters | Model |
|----------------|---------------------|-------|
| Dichotomous (0/1) | 1-parameter | Rasch |
| Dichotomous (0/1) | 2-parameter | 2PL |
| Dichotomous (0/1) | 3-parameter | 3PL (includes guessing) |
| Polytomous | - | GRM, PCM, GPCM |

### 9.2 Item Parameters

| Parameter | Symbol | Meaning | Acceptable Range |
|-----------|--------|---------|-----------------|
| Difficulty | b | Ability level for 50% correct | -3 to +3 |
| Discrimination | a | Slope of the curve | 0.5 to 2.5 |
| Guessing | c | Lower asymptote | 0 to 0.35 |

### 9.3 Model Fit

- **Item fit**: S-chi-squared (p > .05 acceptable)
- **Overall fit**: M2 statistic, RMSEA
- **Relative comparison**: AIC, BIC

---

## 10. Longitudinal Analysis

### 10.1 Method Selection

| Research Question | Recommended Method |
|-------------------|-------------------|
| Group differences over time | Repeated Measures ANOVA / Mixed ANOVA |
| Individual trajectory differences | Latent Growth Model (LGM) |
| Temporal causality between variables | Cross-Lagged Panel Model (CLPM) |
| Separating trait and state effects | Random Intercept Cross-Lagged (RI-CLPM) |
| Trajectory classification of subgroups | Growth Mixture Model (GMM) |

### 10.2 Latent Growth Model (LGM)

```
Measurement Model:
Y_t = η_i + λ_t × η_s + ε_t

η_i = Intercept factor (initial level)
η_s = Slope factor (rate of change)
λ_t = Time coding (0, 1, 2... or freely estimated)
```

### 10.3 Cross-Lagged Panel Model

**Traditional CLPM** issues:
- Confounds between-person differences with within-person change
- Overestimates cross-variable effects

**RI-CLPM** advantages:
- Separates stable traits (random intercepts)
- Captures true within-person temporal change

---

## Method Selection Decision Tree

### Difference Test Decision

```
Comparing mean differences?
    │
    ├── 2 Groups
    │   ├── Independent Samples
    │   │   ├── Normal + equal variance → Independent t-test
    │   │   ├── Normal + unequal variance → Welch's t ⭐Recommended
    │   │   └── Non-normal → Mann-Whitney U
    │   └── Paired Samples
    │       ├── Differences normal → Paired t-test
    │       └── Differences non-normal → Wilcoxon Signed-Rank
    │
    └── 3+ Groups
        ├── Between-subjects design
        │   ├── One-factor → One-way ANOVA / Kruskal-Wallis
        │   └── Multi-factor → Factorial ANOVA
        ├── Within-subjects design → Repeated Measures ANOVA / Friedman
        └── Mixed design → Mixed ANOVA / HLM
```

### Relationship Analysis Decision

```
Exploring variable relationships?
    │
    ├── Prediction/Explanation
    │   ├── DV continuous → Linear Regression
    │   ├── DV dichotomous → Logistic Regression
    │   ├── DV count → Poisson / Negative Binomial
    │   └── DV ordinal → Ordinal Logistic
    │
    ├── Mediation/Moderation
    │   ├── Simple mediation → Bootstrap / PROCESS
    │   ├── Simple moderation → Hierarchical regression + interaction term
    │   └── Complex paths → SEM
    │
    └── Latent Variables
        ├── Scale validation → CFA
        ├── Latent variable relationships → SEM
        └── Longitudinal latent variables → LGM / RI-CLPM
```

### Data Structure Decision

```
Does the data have a nested/hierarchical structure?
    │
    ├── Yes (ICC > .05)
    │   ├── 2-level nesting → HLM
    │   ├── 3+ level nesting → Multi-level HLM
    │   └── Cross-classified → Cross-classified random effects
    │
    └── No
        └── Use single-level analysis methods
```

---

## Appendix: Reporting Standards Quick Reference

### Difference Tests

```
t-test: t(df) = X.XX, p = .XXX, d = X.XX
ANOVA: F(df1, df2) = X.XX, p = .XXX, η² = .XX
```

### Regression Analysis

```
Regression coefficient: B = X.XX, SE = X.XX, β = .XX, t = X.XX, p = .XXX
Model fit: R² = .XX, F(df1, df2) = X.XX, p < .001
```

### SEM

```
Model fit: χ²(df) = X.XX, p = .XXX, CFI = .XX, TLI = .XX,
         RMSEA = .XX [90% CI: .XX, .XX], SRMR = .XX
```

### Meta-Analysis

```
Overall effect: d = X.XX [95% CI: X.XX, X.XX], Z = X.XX, p < .001
Heterogeneity: Q(df) = X.XX, p = .XXX, I² = XX%
```

---

## 11. Medical Research Methods

### 11.1 Reliability and Validity Analysis

**Applicable Scenario**: Scale/questionnaire development and validation

| Measure | Method | Standard | Purpose |
|---------|--------|----------|---------|
| Internal consistency | Cronbach's alpha | > .70 acceptable, > .80 good | Overall scale reliability |
| Composite reliability | McDonald's omega | > .70 acceptable | More accurate reliability estimate |
| Test-retest reliability | ICC or Pearson r | > .70 | Temporal stability |
| Inter-rater reliability | ICC, Kappa | ICC > .75 good | Rater agreement |
| Content validity | CVI (Content Validity Index) | > .78 | Expert review |
| Construct validity | EFA then CFA | Fit indices met | Factor structure |
| Criterion validity | Correlation with gold standard | r significant | External criterion |

**Analysis Workflow**:
```
Standard Scale Validation Procedure:
1. Item analysis (delete items with CITC < .30)
2. EFA to explore factor structure (KMO > .60, Bartlett's significant)
3. CFA to confirm factor structure (CFI > .90, RMSEA < .08)
4. Reliability (Cronbach's alpha > .70)
5. Validity (convergent validity AVE > .50, discriminant validity)
```

### 11.2 Diagnostic Accuracy Analysis

| Measure | Definition | Formula |
|---------|-----------|---------|
| Sensitivity | True positive rate | TP / (TP + FN) |
| Specificity | True negative rate | TN / (TN + FP) |
| Positive Predictive Value (PPV) | Proportion of true positives among positives | TP / (TP + FP) |
| Negative Predictive Value (NPV) | Proportion of true negatives among negatives | TN / (TN + FN) |
| Positive Likelihood Ratio (LR+) | Diagnostic value of a positive result | Se / (1 - Sp) |
| Negative Likelihood Ratio (LR-) | Exclusion value of a negative result | (1 - Se) / Sp |
| AUC | Overall diagnostic accuracy | 0.7-0.8 acceptable, 0.8-0.9 good, >0.9 excellent |
| Youden's J | Optimal cutoff value | Se + Sp - 1 |

**Reporting Standard**:
```
The ROC analysis yielded an AUC of .85 (95% CI [.80, .90]).
At the optimal cutoff of X, sensitivity was .82 and specificity was .78.
```

### 11.3 Survival Analysis

| Method | Applicable Scenario | Assumptions |
|--------|---------------------|-------------|
| Kaplan-Meier | Describe survival function | Independent censoring |
| Log-rank test | Compare survival curves | Proportional hazards |
| Cox Regression | Multivariable prediction | Proportional hazards (PH assumption) |

**PH Assumption Test**: Schoenfeld residuals test (p > .05 satisfied)

**Effect Size**: Hazard Ratio (HR)
- HR > 1: Higher risk in the exposed group
- HR < 1: Lower risk in the exposed group
- HR = 1: No difference

**Reporting Standard**:
```
Cox regression: HR = X.XX (95% CI [X.XX, X.XX]), p = .XXX
K-M median survival time: X months (95% CI [X, X])
```

### 11.4 Agreement Analysis

| Method | Applicable Scenario | Data Type |
|--------|---------------------|-----------|
| Cohen's Kappa | 2 raters | Categorical |
| Fleiss' Kappa | 3+ raters | Categorical |
| ICC | 2+ raters | Continuous |
| Bland-Altman | 2 measurement methods | Continuous |

**Kappa Interpretation**:
| kappa Range | Agreement Level |
|-------------|----------------|
| < 0.20 | Poor |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Excellent |

**ICC Type Selection**:
| Type | Model | Definition | Applicable Scenario |
|------|-------|-----------|---------------------|
| ICC(1,1) | One-way random | Consistency | Each target rated by different raters |
| ICC(2,1) | Two-way random | Consistency | Raters randomly sampled, focus on absolute agreement |
| ICC(3,1) | Two-way mixed | Consistency | Fixed raters, most commonly used |

### 11.5 Sample Size Calculation Quick Reference

| Analysis Method | Small Effect | Medium Effect | Large Effect | Parameters |
|----------------|-------------|---------------|-------------|------------|
| Independent t-test | 394/group | 64/group | 26/group | d=0.2/0.5/0.8 |
| Paired t-test | 199 | 34 | 15 | d=0.2/0.5/0.8 |
| ANOVA (3 groups) | 969/group | 159/group | 66/group | f=0.1/0.25/0.4 |
| Correlation | 783 | 85 | 29 | r=.1/.3/.5 |
| Regression (3 IVs) | 550 | 77 | 36 | f²=.02/.15/.35 |
| Chi-square (2x2) | 785 | 88 | 26 | w=.1/.3/.5 |
| SEM | — | 200-400 | — | Rule of thumb: N > 200 or 10:1 |
| HLM | — | 30+ groups x 30+/group | — | Rule of thumb: number of groups is more important |

> Default parameters: alpha = .05, power = .80, two-tailed test

### 11.6 Propensity Score Matching (PSM)

**Applicable Scenario**: Controlling confounders in observational studies

**Analysis Workflow**:
```
1. Estimate propensity scores (Logistic regression)
2. Select matching method (nearest neighbor / caliper / kernel matching)
3. Check matching quality (SMD < 0.1)
4. Estimate effects after matching
5. Sensitivity analysis
```

**Standardized Mean Difference (SMD)**:
- SMD < 0.1: Well balanced
- SMD 0.1-0.25: Acceptable
- SMD > 0.25: Insufficient balance

---

## Method Selection Decision Tree (Medical Research)

```
Common medical research analysis needs:
    │
    ├── Scale development/validation?
    │   ├── Explore structure → EFA
    │   ├── Confirm structure → CFA
    │   └── Reliability → Cronbach's alpha + ICC
    │
    ├── Diagnostic/screening tool evaluation?
    │   ├── Diagnostic accuracy → ROC/AUC
    │   ├── Cutoff determination → Youden's index
    │   └── Method comparison → Bland-Altman
    │
    ├── Prognosis/survival analysis?
    │   ├── Describe survival → Kaplan-Meier
    │   ├── Group comparison → Log-rank
    │   └── Multivariable prediction → Cox Regression
    │
    ├── Controlling confounders in observational studies?
    │   ├── Known confounders → Multivariable regression
    │   └── Multiple confounders → Propensity score matching
    │
    └── Inter-rater agreement?
        ├── Categorical data → Kappa
        ├── Continuous data → ICC
        └── Method comparison → Bland-Altman
```

---

## Appendix: Reporting Standards Quick Reference (Supplement)

### Reliability and Validity

```
Reliability: Cronbach's alpha = .XX
ICC: ICC(3,1) = .XX, 95% CI [.XX, .XX]
```

### Diagnostic Analysis

```
AUC = .XX (95% CI [.XX, .XX])
Optimal cutoff: X (Youden's J = .XX)
Sensitivity = .XX, Specificity = .XX
```

### Survival Analysis

```
K-M: Median survival time = X months (95% CI [X, X])
Log-rank: χ²(df) = X.XX, p = .XXX
Cox: HR = X.XX (95% CI [X.XX, X.XX]), p = .XXX
```

### Agreement

```
Kappa: κ = .XX (95% CI [.XX, .XX]), p = .XXX
ICC: ICC(3,1) = .XX (95% CI [.XX, .XX])
Bland-Altman: Mean difference = X.XX, LoA = [X.XX, X.XX]
```

---

*Index version: 3.0 | Last updated: 2026-02-09*
*Added: Medical research methods (reliability/validity, diagnostic analysis, survival analysis, agreement, sample size calculation, PSM)*
