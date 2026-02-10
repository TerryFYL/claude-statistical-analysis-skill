---
name: statistical-analysis
description: >
  Statistical analysis service. Trigger conditions: (1) Upload a data file, (2) Say "statistical analysis" / "analyze my data" / "run the data",
  (3) Mention a specific statistical method (t-test / regression / SEM, etc.).

  **Core principle**: Proactively diagnose like a top-tier statistical consultant — don't just execute the method the user requests, ensure the method selection is correct.
---

# Statistical Analysis Service v4

## Core Philosophy: Diagnosis Before Analysis

v3 was "do whatever the user asks"; v4 is "diagnose the data and assumptions first, then decide what to do."

```
v3: User request → Select path → Execute
v4: User request → Data profile → Assumption checks → Intelligent method selection → Execute → Triplet output (table + figure + paragraph)
```

---

## Workflow Overview

```
User request
    |
    v
+-------------------------------------+
|  Step 0: Data Profile (required for  |
|          all paths)                  |
|  - Sample size, variable types,      |
|    missing patterns                  |
|  - Distribution characteristics,     |
|    outlier detection                 |
|  - Completed within 30 seconds,      |
|    no confirmation needed            |
+-------------------------------------+
    |
    v
Assess complexity -> Select path
    |
    |-- Quick path -> Assumption self-check -> Execute -> Triplet output
    |-- Light path -> Assumption self-check -> Confirm variables -> Execute -> Triplet output
    +-- Full path  -> Four stages (incl. assumption checks) -> Triplet output
```

---

## Step 0: Data Profile

**Must be executed before all analyses**. Output format:

```markdown
## Data Profile

| Metric | Value |
|--------|-------|
| Sample size | N = 3248 |
| Variables | 94 (continuous: 60, categorical: 34) |
| Missing rate | Overall 2.3%, highest: Variable X (15.2%) |
| Outliers | Variable Y has 12 (> 3SD) |

### Key Variable Distributions
| Variable | M (SD) | Skewness | Kurtosis | Normality |
|----------|--------|----------|----------|-----------|
| DV_score | 3.45 (1.23) | 0.34 | -0.12 | Pass |
| IV_score | 2.89 (0.98) | 1.45 | 3.21 | Right-skewed |

### Data Alerts
- Warning: IV_score is right-skewed; parametric tests should be used with caution
- Warning: Variable X has 15% missing; recommend checking MCAR/MAR
- OK: Sample size is sufficient for all standard analyses
```

**Execution code** (runs automatically, not displayed to the user):

```python
import pandas as pd
import numpy as np
from scipy import stats

def data_profile(df, target_vars=None):
    """Generate data profile; target_vars are key variables mentioned by the user"""
    vars_to_check = target_vars or df.select_dtypes(include=[np.number]).columns[:10]

    profile = {}
    for var in vars_to_check:
        col = df[var].dropna()
        n = len(col)
        # Normality test: Shapiro-Wilk for n<50, skewness+kurtosis for n>=50
        skew, kurt = col.skew(), col.kurtosis()
        if n < 50:
            _, p_norm = stats.shapiro(col)
            is_normal = p_norm > .05
        else:
            is_normal = abs(skew) < 2 and abs(kurt) < 7

        profile[var] = {
            'M': col.mean(), 'SD': col.std(),
            'missing': df[var].isna().sum(),
            'missing_pct': df[var].isna().mean() * 100,
            'skew': skew, 'kurt': kurt,
            'is_normal': is_normal,
            'outliers_3sd': ((col - col.mean()).abs() > 3 * col.std()).sum()
        }
    return profile
```

---

## Complexity Assessment and Path Selection

| Complexity | Analysis Type | Path | Confirmations |
|------------|---------------|------|---------------|
| **Simple** | Descriptive statistics, t-test, chi-square, correlation, reliability/validity | Quick | 0 |
| **Moderate** | Regression, ANOVA, moderation, mediation, ROC/AUC, survival analysis | Light | 1 |
| **Complex** | SEM/CFA, HLM, IRT, meta-analysis, RI-CLPM, propensity score matching | Full | 3-4 |
| **Planning** | Sample size calculation / Power Analysis (no data, parameters only) | Dedicated | 1 |

---

## Quick Path (Simple Analysis)

**Applicable to**: Descriptive statistics, t-test, chi-square, correlation, reliability/validity

**Flow**: Data profile -> Assumption self-check -> Execute -> Triplet output (table + figure + paragraph)

### Assumption Self-Check (automatic, embedded in execution)

```python
def check_and_run_ttest(df, group_var, value_var, group1, group2):
    """Automatically check assumptions and select the appropriate t-test"""
    g1 = df[df[group_var] == group1][value_var].dropna()
    g2 = df[df[group_var] == group2][value_var].dropna()

    # 1. Normality test
    n1, n2 = len(g1), len(g2)
    if min(n1, n2) < 50:
        _, p1 = stats.shapiro(g1)
        _, p2 = stats.shapiro(g2)
        normal = p1 > .05 and p2 > .05
    else:
        normal = abs(g1.skew()) < 2 and abs(g2.skew()) < 2

    if not normal:
        # Non-parametric alternative
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        method = "Mann-Whitney U"
        effect = abs(stat - n1*n2/2) / (n1*n2)  # rank-biserial r
        return method, stat, p, effect

    # 2. Homogeneity of variance test
    _, p_levene = stats.levene(g1, g2)
    equal_var = p_levene > .05

    # 3. Select t-test type
    stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
    method = "Independent samples t-test" if equal_var else "Welch's t-test"

    # 4. Cohen's d
    pooled_sd = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    d = (g1.mean() - g2.mean()) / pooled_sd

    return method, stat, p, d
```

**Key behavior**: When assumptions are not met, **automatically switch methods and inform the user**:

```markdown
> Note: Variable X failed the normality test (Shapiro-Wilk p = .003).
> Automatically switched to Mann-Whitney U test (non-parametric alternative).
```

---

## Light Path (Moderate Analysis)

**Applicable to**: Regression, ANOVA, moderation effect, mediation effect, ROC/AUC, survival analysis

**Flow**: Data profile -> Assumption self-check -> Confirm variables -> Execute -> Triplet output (table + figure + paragraph)

### Variable Confirmation (Enhanced)

```markdown
Data loaded: N = 3005

Please confirm variable roles:
| Role | Variable | Type | Distribution |
|------|----------|------|--------------|
| Dependent (Y) | SDQ total score | Continuous | Normal |
| Independent (X) | Gaming disorder score | Continuous | Right-skewed (skewness=1.45) |
| Moderator (M) | Only child | Dichotomous | — |
| Controls | Age, gender | Continuous/Dichotomous | OK |

### Automatic Assumption Check Results
- OK: Sample size sufficient (N=3005, well above minimum requirement)
- OK: Multicollinearity: all VIF < 5
- Warning: Independent variable is right-skewed; consider: (a) log transformation (b) robust standard errors (c) keep as-is
- Recommendation: Use robust standard errors (HC3) to preserve original variable meaning

Confirm to proceed with analysis.
```

---

## Full Path (Complex Analysis)

**Applicable to**: SEM/CFA, HLM, IRT, meta-analysis, RI-CLPM

**Flow**: Four stages, confirmation at each stage (see `references/full-workflow.md`)

```
Stage 1: Data profile + cleaning plan -> Pause for confirmation
Stage 2: Execute cleaning + assumption checks -> Pause for confirmation
Stage 3: Analysis plan + sample size adequacy -> Pause for confirmation
Stage 4: Execute analysis -> Triplet output (table + figure + paragraph)
```

---

## Power Analysis Path (Sample Size Calculation)

**Trigger**: User says "sample size" / "statistical power" / "power analysis" / "how many participants"

**No data file needed**, only parameters:

```markdown
## Sample Size Calculation

Please provide the following information:
| Parameter | Your setting | Default |
|-----------|-------------|---------|
| Analysis method | ? | — |
| Expected effect size | ? | Medium (d=0.5 / f=0.25 / r=.30) |
| Significance level (alpha) | ? | .05 |
| Statistical power (1-beta) | ? | .80 |
| Number of groups | ? | 2 |
| One-tailed? | ? | Two-tailed |
```

**Execution**:
```python
from scipy import stats
import numpy as np

def power_ttest(d, alpha=0.05, power=0.80, ratio=1, alternative='two-sided'):
    """t-test sample size calculation (per group)"""
    from scipy.optimize import brentq
    def power_func(n):
        df = (1+ratio)*n - 2
        nc = d * np.sqrt(n*ratio/(1+ratio))  # noncentrality
        if alternative == 'two-sided':
            crit = stats.t.ppf(1 - alpha/2, df)
            p = 1 - stats.nct.cdf(crit, df, nc) + stats.nct.cdf(-crit, df, nc)
        else:
            crit = stats.t.ppf(1 - alpha, df)
            p = 1 - stats.nct.cdf(crit, df, nc)
        return p - power
    n = int(np.ceil(brentq(power_func, 2, 10000)))
    return n

# Quick reference table for common methods
power_table = {
    't-test': {'Small (d=0.2)': 394, 'Medium (d=0.5)': 64, 'Large (d=0.8)': 26},
    'ANOVA (3 groups)': {'Small (f=0.1)': 969, 'Medium (f=0.25)': 159, 'Large (f=0.4)': 66},
    'Correlation': {'Small (r=.1)': 783, 'Medium (r=.3)': 85, 'Large (r=.5)': 29},
    'Regression (3 IVs)': {'Small (f2=.02)': 550, 'Medium (f2=.15)': 77, 'Large (f2=.35)': 36},
}
```

---

## APA Result Paragraph Generation (Final Output for All Paths)

**After each analysis is completed, in addition to tables and figures, a result paragraph ready for direct insertion into a manuscript must be generated.**

### Result Paragraph Templates

**t-test**:
> An independent samples t-test revealed a significant difference in {DV} between {group1} (M = {m1}, SD = {sd1}) and {group2} (M = {m2}, SD = {sd2}), t({df}) = {t}, p {p_text}, Cohen's d = {d}. The effect size was {small/medium/large}.

**Correlation analysis**:
> Pearson correlation analysis showed that {var1} was significantly {positively/negatively} correlated with {var2}, r({df}) = {r}, p {p_text}. The correlation coefficient indicated a {small/medium/large} effect.

**Regression analysis**:
> A hierarchical multiple regression was conducted. In Step 1, {controls} were entered, accounting for {R1²}% of variance in {DV}, F({df1}, {df2}) = {F1}, p {p1}. In Step 2, {predictor} was added, significantly improving model fit, ΔR² = {dr2}, ΔF({ddf1}, {ddf2}) = {dF}, p {dp}. {Predictor} was a significant predictor (β = {beta}, p {p_text}).

**Mediation effect**:
> A mediation analysis using 5000 bootstrap samples revealed a significant indirect effect of {X} on {Y} through {M} (indirect effect = {ab}, 95% CI [{ci_lo}, {ci_hi}]). The direct effect was {significant/non-significant} (c' = {c_prime}, p {p_text}), suggesting {full/partial} mediation.

**Moderation effect**:
> The interaction between {X} and {W} was significant (B = {b3}, SE = {se}, p {p_text}), indicating that {W} moderated the relationship between {X} and {Y}. Simple slope analysis showed that the effect of {X} on {Y} was significant at high levels of {W} (+1SD: B = {b_high}, p {p_high}) but not at low levels (-1SD: B = {b_low}, p {p_low}).

### Result Paragraph Code

```python
def format_p(p):
    """APA-formatted p value"""
    if p < .001: return "< .001"
    return f"= {p:.3f}"

def effect_size_label(d):
    """Cohen's d effect size descriptor"""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

def write_ttest_result(group1_name, group2_name, dv_name,
                        m1, sd1, m2, sd2, t, df, p, d, method="Independent samples t-test"):
    """Generate t-test result paragraph"""
    sig = "significant" if p < .05 else "non-significant"
    return (
        f"An {method.lower()} revealed a {sig} difference in {dv_name} "
        f"between {group1_name} (M = {m1:.2f}, SD = {sd1:.2f}) and "
        f"{group2_name} (M = {m2:.2f}, SD = {sd2:.2f}), "
        f"t({df}) = {t:.2f}, p {format_p(p)}, Cohen's d = {d:.2f}. "
        f"The effect size was {effect_size_label(d)}."
    )
```

**Output language**: English by default (standard for manuscripts). If the user says "Chinese", generate a Chinese version.

---

## Automatic Figure Generation Engine (Required for All Paths)

**Core rule**: After each analysis is executed, corresponding figures must be automatically generated. Figures are not optional attachments — they are a required component of the output triplet.

### Initialization (automatically executed before each plot)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def init_figure(figsize=(10, 6)):
    """Unified figure initialization: CJK fonts + APA style"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
```

### Analysis Type to Figure Function Mapping

| Analysis Type | Figure Function | Output Filename |
|---------------|-----------------|-----------------|
| Descriptive statistics | `plot_distribution()` | `fig_distribution.png` |
| Group comparison (t-test) | `plot_group_comparison()` | `fig_group_comparison.png` |
| Correlation analysis | `plot_correlation_heatmap()` | `fig_correlation_heatmap.png` |
| Regression analysis | `plot_regression_coefficients()` | `fig_regression_coef.png` |
| Hierarchical regression | `plot_hierarchical_regression()` | `fig_hierarchical_reg.png` |
| ANOVA | `plot_anova_boxplot()` | `fig_anova_boxplot.png` |
| Moderation effect | `plot_moderation_interaction()` | `fig_moderation.png` |
| Mediation effect | `plot_mediation_path()` | `fig_mediation_path.png` |
| ROC/AUC | `plot_roc_curve()` | `fig_roc_curve.png` |
| Survival analysis | `plot_km_curve()` | `fig_km_curve.png` |

### Figure Code Templates

**Group comparison bar chart** (used after t-test/chi-square):
```python
def plot_group_comparison(means, sds, group_labels, dv_labels, title, save_path,
                           p_values=None, figsize=(10, 6)):
    """Grouped bar chart with error bars"""
    fig, ax = init_figure(figsize)
    x = np.arange(len(dv_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, means[0], width, yerr=sds[0], capsize=4,
                   color='#4C72B0', alpha=0.85, label=group_labels[0])
    bars2 = ax.bar(x + width/2, means[1], width, yerr=sds[1], capsize=4,
                   color='#DD8452', alpha=0.85, label=group_labels[1])
    # Significance stars
    if p_values is not None:
        for i, p in enumerate(p_values):
            if p < .001: star = '***'
            elif p < .01: star = '**'
            elif p < .05: star = '*'
            else: star = 'ns'
            max_y = max(means[0][i] + sds[0][i], means[1][i] + sds[1][i])
            ax.text(x[i], max_y * 1.05, star, ha='center', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(dv_labels, rotation=15, ha='right')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**Correlation heatmap**:
```python
def plot_correlation_heatmap(corr_matrix, p_matrix, title, save_path, figsize=(12, 10)):
    """Correlation heatmap with significance stars"""
    fig, ax = init_figure(figsize)
    # Generate star annotations
    annot = corr_matrix.round(2).astype(str)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            p = p_matrix.iloc[i, j]
            r = corr_matrix.iloc[i, j]
            star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            annot.iloc[i, j] = f"{r:.2f}{star}"
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**Regression coefficient forest plot**:
```python
def plot_regression_coefficients(names, betas, ci_lower, ci_upper, title, save_path,
                                  figsize=(8, 6)):
    """Standardized regression coefficient forest plot (with 95% CI)"""
    fig, ax = init_figure(figsize)
    y_pos = np.arange(len(names))
    colors = ['#C44E52' if b > 0 else '#4C72B0' for b in betas]
    ax.barh(y_pos, betas, color=colors, alpha=0.7, height=0.6)
    ax.errorbar(betas, y_pos, xerr=[np.array(betas)-np.array(ci_lower),
                np.array(ci_upper)-np.array(betas)],
                fmt='none', color='black', capsize=3, linewidth=1.2)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Standardized Regression Coefficient (beta)')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**Hierarchical regression dual-panel figure**:
```python
def plot_hierarchical_regression(step_labels, r2_values, delta_r2, predictor_names,
                                  betas, title, save_path, figsize=(14, 6)):
    """Left: Stacked Delta-R-squared bars | Right: Beta coefficient horizontal bars"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # Left panel: Stacked Delta-R-squared
    bottom = 0
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for i, (label, dr2) in enumerate(zip(step_labels, delta_r2)):
        ax1.bar(0, dr2, bottom=bottom, color=colors[i % len(colors)],
                alpha=0.85, label=f'{label} (Delta-R2={dr2:.3f})')
        bottom += dr2
    ax1.set_ylabel('R-squared')
    ax1.set_title('Model Explained Variance', fontweight='bold')
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_xticks([])
    # Right panel: Beta coefficients
    y_pos = np.arange(len(predictor_names))
    colors_beta = ['#C44E52' if b > 0 else '#4C72B0' for b in betas]
    ax2.barh(y_pos, betas, color=colors_beta, alpha=0.7, height=0.6)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictor_names)
    ax2.set_xlabel('Standardized Coefficient (beta)')
    ax2.set_title('Predictor Contributions', fontweight='bold')
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**Moderation interaction plot**:
```python
def plot_moderation_interaction(x_vals, y_low, y_high, x_label, y_label,
                                 mod_label, title, save_path, figsize=(8, 6)):
    """Simple slopes plot: X-Y relationship at high/low levels of the moderator"""
    fig, ax = init_figure(figsize)
    ax.plot(x_vals, y_low, 'o-', color='#4C72B0', label=f'{mod_label} -1SD', linewidth=2)
    ax.plot(x_vals, y_high, 's-', color='#C44E52', label=f'{mod_label} +1SD', linewidth=2)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**Mediation path diagram**:
```python
def plot_mediation_path(x_name, m_name, y_name, a, b, c, c_prime, indirect,
                         ci_lo, ci_hi, title, save_path, figsize=(10, 6)):
    """Mediation path diagram: X->M->Y + direct effect"""
    fig, ax = init_figure(figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    # Boxes
    for (cx, cy, label) in [(1.5, 3, x_name), (5, 5, m_name), (8.5, 3, y_name)]:
        ax.add_patch(plt.Rectangle((cx-1, cy-0.4), 2, 0.8,
                     fill=True, facecolor='#E8E8E8', edgecolor='black', linewidth=1.5))
        ax.text(cx, cy, label, ha='center', va='center', fontsize=11, fontweight='bold')
    # Arrows + coefficients
    def sig_star(p):
        if p < .001: return '***'
        if p < .01: return '**'
        if p < .05: return '*'
        return ''
    ax.annotate('', xy=(4, 5), xytext=(2.5, 3.4),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(2.8, 4.4, f'a = {a:.3f}', fontsize=10)
    ax.annotate('', xy=(7.5, 3.4), xytext=(6, 5),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(7, 4.4, f'b = {b:.3f}', fontsize=10)
    ax.annotate('', xy=(7.5, 3), xytext=(2.5, 3),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(5, 2.3, f"c' = {c_prime:.3f}", fontsize=10, ha='center')
    ax.text(5, 1.2, f'Indirect effect = {indirect:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**ANOVA box plot**:
```python
def plot_anova_boxplot(df, group_var, dv_vars, title, save_path, figsize=(14, 8)):
    """Multi-dependent-variable grouped box plot grid"""
    n_vars = len(dv_vars)
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    palette = sns.color_palette("Set2", df[group_var].nunique())
    for i, dv in enumerate(dv_vars):
        sns.boxplot(data=df, x=group_var, y=dv, palette=palette, ax=axes[i])
        axes[i].set_title(dv, fontweight='bold')
        sns.despine(ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## Output Triplet Check (Automatically Executed at End of Each Analysis)

**Rule**: After each analysis is completed, a self-check must verify that the output triplet is complete. If any item is missing, it is automatically generated.

```
Analysis complete
    |
    v
Check output triplet:
    |-- APA table (Excel/Markdown) -> Generated
    |-- Figure (PNG, dpi=300) -> Generated
    +-- Result paragraph -> Generated
           |
           +-- All present -> Output complete
           +-- Missing item -> Automatically generate missing items
```

**Self-check code** (embedded at the end of the analysis flow):
```python
def check_output_triplet(analysis_type, outputs):
    """Check whether the output triplet is complete"""
    required = ['table', 'figure', 'paragraph']
    missing = [r for r in required if r not in outputs or outputs[r] is None]
    if missing:
        print(f"Warning: {analysis_type} missing: {', '.join(missing)}. Auto-generating...")
    return missing
```

**Behavioral constraints**:
- When `check_output_triplet()` returns a non-empty list, **missing items must be generated immediately**
- It is not acceptable to skip figures on the grounds that "the user didn't ask for a figure"
- The figure type is automatically determined by the "Analysis Type to Figure Function Mapping" table

---

## Medical Research Methods

### Reliability and Validity Analysis (Quick Path)

**Trigger**: "reliability" / "validity" / "Cronbach" / "scale validation" / "internal consistency"

```python
import pingouin as pg

# Cronbach's alpha
alpha = pg.cronbach_alpha(df[items])

# If alpha < .70 -> suggest examining individual items
if alpha[0] < .70:
    # Alpha after deleting each item
    for item in items:
        remaining = [i for i in items if i != item]
        a = pg.cronbach_alpha(df[remaining])
        print(f"Alpha after deleting {item} = {a[0]:.3f}")
```

**Output**: Alpha coefficient + Corrected Item-Total Correlation (CITC) + alpha-if-item-deleted table

### ROC/AUC Analysis (Light Path)

**Trigger**: "ROC" / "AUC" / "diagnostic accuracy" / "sensitivity" / "specificity" / "cutoff value"

```python
from sklearn.metrics import roc_curve, auc, confusion_matrix

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Optimal cutoff (Youden's index)
youden = tpr - fpr
optimal_idx = np.argmax(youden)
optimal_threshold = thresholds[optimal_idx]

# Metrics at optimal cutoff
y_pred = (y_scores >= optimal_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
```

**Output**: ROC curve figure + AUC + optimal cutoff + sensitivity/specificity/PPV/NPV table

### Survival Analysis (Light Path)

**Trigger**: "survival analysis" / "Kaplan-Meier" / "Cox" / "hazard ratio" / "HR" / "survival curve"

Code templates available in `references/code-patterns.md` -> Survival Analysis section

### Intraclass Correlation Coefficient ICC (Quick Path)

**Trigger**: "ICC" / "inter-rater agreement" / "inter-rater reliability" / "intraclass correlation"

```python
import pingouin as pg

# ICC(3,1) - most common: two-way mixed, consistency, single measures
icc = pg.intraclass_corr(data=df_long, targets='subject',
                          raters='rater', ratings='score')
# Report the ICC(3,1) row
```

---

## Execution Strategy

### Default: Python (install missing packages as needed)

```bash
pip3 install scipy statsmodels pingouin matplotlib seaborn scikit-learn --quiet --break-system-packages
```

**CJK plotting support**:
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
```

### Method-Library Quick Reference

| Method | Library | Example |
|--------|---------|---------|
| Descriptive statistics | pandas | `df.describe()` |
| t-test | scipy | `stats.ttest_ind(a, b)` |
| Chi-square test | scipy | `stats.chi2_contingency(table)` |
| Correlation analysis | pandas/scipy | `df.corr()` / `pearsonr()` |
| Regression | statsmodels | `sm.OLS(y, X).fit()` |
| ANOVA | pingouin | `pg.anova(data, dv, between)` |
| Moderation effect | statsmodels | Interaction term regression |
| Mediation effect | Custom | Bootstrap (see code-patterns) |
| Reliability/Validity | pingouin | `pg.cronbach_alpha()` |
| ROC/AUC | sklearn | `roc_curve()` / `auc()` |
| Survival analysis | lifelines | `KaplanMeierFitter` / `CoxPHFitter` |
| ICC | pingouin | `pg.intraclass_corr()` |
| Power | scipy | Custom function (see above) |

### Only When User Requests or When Necessary: R Docker

**Trigger conditions**:
- User explicitly says "use R" / "use lavaan" / "use lme4"
- The analysis method cannot be implemented in Python (SEM path diagrams, HLM random effects plots)

```bash
# R Docker location
cd docker/

# Quick usage
./r-stat.sh build    # First-time build
./r-stat.sh run x.R  # Execute script
```

**Pre-installed R packages**: lavaan, lme4, metafor, mirt, psych, tidyverse, semPlot, effectsize, parameters, performance

---

## Output Specifications

### Required Output Triplet

Each analysis must output:
1. **APA table** (Excel + Markdown) -- format per `references/table-formats.md`
2. **Figure** (PNG, dpi=300) -- type per table below
3. **Result paragraph** (English, switchable to Chinese) -- ready for direct insertion into a manuscript

### Table Format (APA 7th Edition)

| Item | Standard |
|------|----------|
| Test statistics | 2 decimal places |
| p values | 3 decimal places; <.001 written as "<.001" |
| Significance | *p<.05, **p<.01, ***p<.001 |
| Effect sizes | Cohen's d, eta-squared, R-squared, f-squared |
| Confidence intervals | 95% CI [lower, upper] |

### Default Figure Output

| Analysis Type | Default Figure |
|---------------|----------------|
| Descriptive statistics | Distribution/box plot |
| Correlation analysis | Heatmap |
| Regression | Coefficient forest plot |
| Moderation effect | Simple slopes plot |
| Mediation effect | Path diagram |
| ROC | ROC curve (with AUC) |
| Survival analysis | Kaplan-Meier survival curve |
| SEM/CFA | Path diagram |
| HLM | Individual trajectory plot |
| Meta-analysis | Forest plot + funnel plot |

---

## Missing Data Handling Strategy

When missing rate > 5%, automatically alert and suggest a handling strategy:

| Missing Rate | Recommended Strategy | Explanation |
|-------------|---------------------|-------------|
| < 5% | Listwise deletion | Minimal impact, simple approach |
| 5-20% | Multiple imputation (MICE) | Preserves sample size, reduces bias |
| > 20% | Check MCAR first, then decide | Requires Little's MCAR test |

```python
# Little's MCAR test (approximate)
# If significant -> data is not missing completely at random; handle with caution
from sklearn.impute import SimpleImputer
# Recommended: use multiple imputation and report sensitivity analysis
```

---

## Resource Files

| File | Purpose |
|------|---------|
| `references/methods-index.md` | Method selection decision tree (incl. medical-specific methods) |
| `references/code-patterns.md` | Code templates (Python + R) |
| `references/full-workflow.md` | Full path detailed workflow |
| `references/table-formats.md` | APA table format templates |
| `docker/` | R Docker execution environment |
| `docker/examples/` | SEM/HLM/meta-analysis/IRT examples |

---

## Changelog

- **v4.1** (2026-02-09): Mandatory output triplet
  - **Fix**: Figure generation upgraded from "specification description" to "in-flow embedded" -- all path endpoints now produce the output triplet
  - Added **automatic figure generation engine**: executable plotting code templates for 8 analysis types
  - Added **output triplet check mechanism**: self-check for table + figure + paragraph at end of analysis; missing items auto-generated
  - Added **init_figure()**: unified CJK font / DPI / APA style initialization
  - Added **analysis type to figure function mapping table**: default figure types for 10 analysis types
  - Fix: Quick/Light/Full path flow descriptions all end with "triplet output"
- **v4.0** (2026-02-09): Diagnosis before analysis
  - Added **Step 0 Data Profile**: automatic data quality check before all analyses
  - Added **assumption self-check engine**: automatically checks prerequisites; switches methods when assumptions are not met
  - Added **APA result paragraph generation**: outputs result descriptions ready for direct manuscript insertion
  - Added **medical-specific methods**: reliability/validity analysis, ROC/AUC, survival analysis routing, ICC, Power Analysis
  - Added **missing data strategy**: automatic handling recommendations based on missing rate
  - Added **Power Analysis dedicated path**: independent workflow for sample size calculation
  - Upgraded **complexity assessment table**: added reliability/validity (simple), ROC/survival analysis (moderate)
  - Upgraded **light path confirmation**: displays assumption check results and data alerts
  - Core philosophy upgraded from "select path by complexity" to "diagnosis before analysis"
- **v3.0** (2026-02-02): Three-tier path system
  - Quick/Light/Full three-tier paths
  - R Docker execution environment
  - APA table format specifications
