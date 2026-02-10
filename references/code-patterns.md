# Common Analysis Code Patterns

> **Note**: This file provides common templates as **reference only**. For methods not covered here, implement directly using appropriate libraries.

## Setup

### Basic Setup (Always Available)

```python
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
```

### Advanced Setup (Install If Needed)

```python
# Install command (run if package not available):
# !pip install semopy factor_analyzer pingouin pymare lifelines pymer4 girth

# Factor Analysis
# from factor_analyzer import FactorAnalyzer, calculate_kmo
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# SEM
# import semopy

# Advanced Statistics (simplified API)
# import pingouin as pg

# Meta-Analysis
# import pymare

# Survival Analysis
# from lifelines import KaplanMeierFitter, CoxPHFitter

# Mixed Effects / HLM
# from pymer4.models import Lmer

# IRT
# import girth
```

### Package Installation Helper

```python
def ensure_package(package_name, import_name=None):
    """Install package if not available."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', package_name])
        __import__(import_name)

# Example usage:
# ensure_package('semopy')
# ensure_package('factor_analyzer')
```

## Chinese Font Configuration (CRITICAL)

**Must run this before any plotting to avoid blank boxes in Chinese labels:**

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_chinese_font():
    """Configure matplotlib for Chinese display. Call before any plotting."""
    # Priority order: try common Chinese fonts
    font_candidates = [
        'Noto Sans CJK JP',      # Available in most Linux systems
        'Noto Sans CJK SC',      # Simplified Chinese specific
        'Noto Sans CJK TC',      # Traditional Chinese
        'Source Han Sans',       # Adobe's open source
        'WenQuanYi Micro Hei',   # Common Linux Chinese font
        'SimHei',                # Windows
        'Microsoft YaHei',       # Windows
        'PingFang SC',           # macOS
        'Heiti SC',              # macOS
    ]

    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    # Find first available font
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
        print(f"Using Chinese font: {selected_font}")
    else:
        print("Warning: No Chinese font found, text may display as boxes")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False
    return selected_font

# Call at start of analysis
setup_chinese_font()
```

**Quick version (if you know Noto is available):**

```python
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

## Descriptive Statistics

```python
# Continuous variables
desc = df[vars].describe().T
desc['skewness'] = df[vars].skew()
desc['kurtosis'] = df[vars].kurtosis()

# Categorical variables
df[var].value_counts()
df[var].value_counts(normalize=True) * 100
```

## Correlation Analysis

```python
# Correlation matrix
corr = df[vars].corr(method='pearson')  # or 'spearman'

# With p-values
from scipy.stats import pearsonr, spearmanr

def corr_with_p(df, vars, method='pearson'):
    n = len(vars)
    corr_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if method == 'pearson':
                r, p = pearsonr(df[vars[i]].dropna(), df[vars[j]].dropna())
            else:
                r, p = spearmanr(df[vars[i]].dropna(), df[vars[j]].dropna())
            corr_mat[i,j], p_mat[i,j] = r, p
    return pd.DataFrame(corr_mat, index=vars, columns=vars), pd.DataFrame(p_mat, index=vars, columns=vars)
```

## Regression Analysis

```python
# Linear regression
X = sm.add_constant(df[ivs])
model = sm.OLS(df[dv], X).fit()
print(model.summary())

# Hierarchical regression
# Step 1: controls only
X1 = sm.add_constant(df[controls])
m1 = sm.OLS(df[dv], X1).fit()

# Step 2: add predictors
X2 = sm.add_constant(df[controls + predictors])
m2 = sm.OLS(df[dv], X2).fit()

# R² change
r2_change = m2.rsquared - m1.rsquared

# Logistic regression
model = sm.Logit(df[dv], X).fit()
```

## Moderation Analysis

```python
# Center variables
df['X_c'] = df['X'] - df['X'].mean()
df['M_c'] = df['M'] - df['M'].mean()
df['XM'] = df['X_c'] * df['M_c']

# Regression with interaction
X = sm.add_constant(df[['X_c', 'M_c', 'XM']])
model = sm.OLS(df['Y'], X).fit()

# Simple slopes (+1SD, -1SD)
m_high = df['M'].mean() + df['M'].std()
m_low = df['M'].mean() - df['M'].std()

b0, b1, b2, b3 = model.params
slope_high = b1 + b3 * (m_high - df['M'].mean())
slope_low = b1 + b3 * (m_low - df['M'].mean())
```

## Mediation Analysis (Bootstrap)

```python
def bootstrap_mediation(df, X, M, Y, n_boot=5000):
    """Bootstrap mediation analysis"""
    n = len(df)
    indirect_effects = []

    for _ in range(n_boot):
        sample = df.sample(n, replace=True)

        # Path a: X -> M
        Xa = sm.add_constant(sample[X])
        a = sm.OLS(sample[M], Xa).fit().params[X]

        # Path b: M -> Y (controlling X)
        Xb = sm.add_constant(sample[[X, M]])
        b = sm.OLS(sample[Y], Xb).fit().params[M]

        indirect_effects.append(a * b)

    indirect = np.array(indirect_effects)
    ci_low, ci_high = np.percentile(indirect, [2.5, 97.5])

    return {
        'indirect_effect': np.mean(indirect),
        'se': np.std(indirect),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': not (ci_low <= 0 <= ci_high)
    }
```

## Visualization

```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Moderation plot
fig, ax = plt.subplots(figsize=(8, 6))
x_range = np.linspace(df['X'].min(), df['X'].max(), 100)

# High moderator
y_high = b0 + slope_high * x_range
ax.plot(x_range, y_high, 'b-', label=f'High M (+1SD)')

# Low moderator
y_low = b0 + slope_low * x_range
ax.plot(x_range, y_low, 'r--', label=f'Low M (-1SD)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.savefig('moderation_plot.png', dpi=300, bbox_inches='tight')
```

## Export to Excel

```python
# Single table
df.to_excel('table.xlsx', index=False)

# Multiple tables in one file
with pd.ExcelWriter('results.xlsx') as writer:
    desc.to_excel(writer, sheet_name='Descriptives')
    corr.to_excel(writer, sheet_name='Correlations')
    reg_table.to_excel(writer, sheet_name='Regression')
```

---

## Advanced Methods (Python)

### Factor Analysis (EFA)

```python
from factor_analyzer import FactorAnalyzer, calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# KMO and Bartlett's test
kmo_all, kmo_model = calculate_kmo(df[vars])
chi_square, p_value = calculate_bartlett_sphericity(df[vars])

print(f"KMO: {kmo_model:.3f}")
print(f"Bartlett's chi-square: {chi_square:.2f}, p = {p_value:.3f}")

# Determine number of factors (parallel analysis or scree plot)
fa = FactorAnalyzer(n_factors=len(vars), rotation=None)
fa.fit(df[vars])
ev, v = fa.get_eigenvalues()

# EFA with rotation
fa = FactorAnalyzer(n_factors=3, rotation='varimax')  # or 'promax' for oblique
fa.fit(df[vars])

# Factor loadings
loadings = pd.DataFrame(
    fa.loadings_,
    index=vars,
    columns=[f'Factor{i+1}' for i in range(3)]
)

# Variance explained
variance = fa.get_factor_variance()
print(f"Variance explained: {variance[1]}")  # Proportional variance
```

### SEM (semopy)

```python
import semopy

# Model specification (lavaan-like syntax)
model_spec = """
# Measurement model
Latent1 =~ x1 + x2 + x3
Latent2 =~ y1 + y2 + y3

# Structural model
Latent2 ~ Latent1
"""

# Fit model
model = semopy.Model(model_spec)
model.fit(df)

# Parameter estimates
print(model.inspect())

# Fit indices
stats = semopy.calc_stats(model)
print(f"CFI: {stats['CFI']:.3f}")
print(f"TLI: {stats['TLI']:.3f}")
print(f"RMSEA: {stats['RMSEA']:.3f}")
print(f"SRMR: {stats['SRMR']:.3f}")
```

### Meta-Analysis

```python
import numpy as np

# Simple random-effects meta-analysis (manual implementation)
def random_effects_meta(effects, variances):
    """
    Random-effects meta-analysis using DerSimonian-Laird method.

    Args:
        effects: array of effect sizes
        variances: array of within-study variances
    """
    weights = 1 / variances

    # Fixed effect estimate
    fe_estimate = np.sum(weights * effects) / np.sum(weights)

    # Q statistic
    Q = np.sum(weights * (effects - fe_estimate)**2)
    df = len(effects) - 1

    # Tau-squared (between-study variance)
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0, (Q - df) / C)

    # Random effects weights and estimate
    re_weights = 1 / (variances + tau2)
    re_estimate = np.sum(re_weights * effects) / np.sum(re_weights)
    re_se = np.sqrt(1 / np.sum(re_weights))

    # 95% CI
    ci_low = re_estimate - 1.96 * re_se
    ci_high = re_estimate + 1.96 * re_se

    return {
        'estimate': re_estimate,
        'se': re_se,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'tau2': tau2,
        'Q': Q,
        'I2': max(0, (Q - df) / Q * 100) if Q > 0 else 0
    }

# Example usage
effects = np.array([0.3, 0.5, 0.2, 0.4, 0.35])
variances = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
result = random_effects_meta(effects, variances)
print(f"Pooled effect: {result['estimate']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
print(f"I-squared: {result['I2']:.1f}%")
```

### Survival Analysis

```python
from lifelines import KaplanMeierFitter, CoxPHFitter

# Kaplan-Meier survival curve
kmf = KaplanMeierFitter()
kmf.fit(df['duration'], event_observed=df['event'], label='Overall')

# Plot survival curve
kmf.plot_survival_function()
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curve')
plt.savefig('km_curve.png', dpi=300, bbox_inches='tight')

# Compare groups
for group in df['group'].unique():
    mask = df['group'] == group
    kmf.fit(df.loc[mask, 'duration'], df.loc[mask, 'event'], label=group)
    kmf.plot_survival_function()

# Cox Proportional Hazards
cph = CoxPHFitter()
cph.fit(df[['duration', 'event', 'age', 'treatment', 'stage']],
        duration_col='duration', event_col='event')

cph.print_summary()
cph.plot()
plt.savefig('cox_forest.png', dpi=300, bbox_inches='tight')
```

### Bayesian Regression (PyMC)

```python
import pymc as pm
import arviz as az

with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=len(predictors))
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Linear model
    mu = alpha + pm.math.dot(df[predictors].values, beta)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df['y'])

    # Sampling
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Summary
print(az.summary(trace, var_names=['alpha', 'beta', 'sigma']))

# Posterior plots
az.plot_posterior(trace, var_names=['beta'])
plt.savefig('posterior.png', dpi=300, bbox_inches='tight')
```

### Time Series (ARIMA)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Stationarity test
result = adfuller(df['y'])
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

# Fit ARIMA model
model = ARIMA(df['y'], order=(1, 1, 1))  # (p, d, q)
results = model.fit()

print(results.summary())

# Forecast
forecast = results.forecast(steps=12)
print(forecast)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
df['y'].plot(ax=ax, label='Observed')
forecast.plot(ax=ax, label='Forecast')
plt.legend()
plt.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
```

---

## R Language Support (Three-tier Execution Strategy)

> **Execution priority**: Python -> Docker R (automatic) -> Output .R file (manual)

### Strategy Description

```
When R execution is required:
1. Check if Docker R environment is available
2. If available -> Execute R code in Docker container
3. If not available -> Generate .R file for user to run in RStudio
```

---

### Docker R Execution (Preferred Approach)

#### Docker Environment Setup (First-time Configuration)

**Dockerfile** (`docker/Dockerfile`):
```dockerfile
FROM rocker/tidyverse:4.3.2

# Chinese language support
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*

# R package installation
RUN Rscript -e 'install.packages(c( \
    "lavaan", "semPlot", "lme4", "lmerTest", \
    "metafor", "mirt", "brms", "survival", "survminer", \
    "psych", "car", "ggpubr", "corrplot", "openxlsx" \
), repos="https://cloud.r-project.org")'

WORKDIR /workspace
CMD ["R"]
```

**Build command**:
```bash
cd docker
docker build -t r-stats-env .
```

#### Docker R Execution Function

```python
import subprocess
import shutil
import tempfile
import os

def check_docker_r_available():
    """Check if Docker R environment is available"""
    if not shutil.which('docker'):
        return False, "Docker is not installed"

    try:
        result = subprocess.run(['docker', 'info'],
                                capture_output=True, timeout=5)
        if result.returncode != 0:
            return False, "Docker is not running"
    except:
        return False, "Docker connection failed"

    try:
        result = subprocess.run(
            ['docker', 'images', '-q', 'r-stats-env'],
            capture_output=True, text=True, timeout=10
        )
        if not result.stdout.strip():
            return False, "r-stats-env image has not been built"
    except:
        return False, "Image check failed"

    return True, "Docker R environment is available"


def execute_r_in_docker(r_code: str, data_file: str = None,
                        output_dir: str = None):
    """
    Execute R code in a Docker container

    Args:
        r_code: R code string
        data_file: Data file path
        output_dir: Output directory

    Returns:
        dict: {'success': bool, 'output': str, 'files': list}
    """
    if output_dir is None:
        output_dir = os.environ.get('OUTPUT_DIR', './outputs')

    os.makedirs(output_dir, exist_ok=True)
    workspace_dir = os.path.dirname(os.path.abspath(data_file)) if data_file else output_dir

    # Write temporary R script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R',
                                      delete=False, dir=workspace_dir,
                                      encoding='utf-8') as f:
        f.write(r_code)
        script_path = f.name
        script_name = os.path.basename(script_path)

    try:
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{os.path.abspath(workspace_dir)}:/workspace',
            '-v', f'{os.path.abspath(output_dir)}:/output',
            '-w', '/workspace',
            'r-stats-env',
            'Rscript', f'/workspace/{script_name}'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr if result.returncode != 0 else None,
            'files': os.listdir(output_dir)
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'output': '', 'errors': 'Execution timed out (300s)'}
    except Exception as e:
        return {'success': False, 'output': '', 'errors': str(e)}
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
```

#### Docker R Usage Example

```python
# Example: Execute lavaan SEM in Docker
r_code = '''
library(lavaan)

# Read data (container path)
data <- read.csv("/workspace/data.csv")

# SEM model
model <- '
  latent1 =~ x1 + x2 + x3
  latent2 =~ y1 + y2 + y3
  latent2 ~ latent1
'

fit <- sem(model, data = data)

# Output results
sink("/output/sem_results.txt")
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("SEM analysis complete! Results saved to /output/sem_results.txt\\n")
'''

# Check and execute
docker_ok, msg = check_docker_r_available()
if docker_ok:
    result = execute_r_in_docker(r_code, data_file="./data.csv")
    if result['success']:
        print("Docker R execution successful")
        print(result['output'])
    else:
        print(f"Execution failed: {result['errors']}")
else:
    print(f"Warning: Docker not available ({msg}), falling back to .R file output")
    # Call generate_r_script() ...
```

---

### Output .R File (Fallback Approach)

When Docker R is not available, generate a standalone .R file for the user to run locally in RStudio.

#### R Code File Generation Function

```python
import os
from datetime import datetime

def generate_r_script(analysis_name: str, r_code: str, data_file: str,
                      required_packages: list, output_dir: str = None):
    """
    Generate a standalone R script file for local execution

    Args:
        analysis_name: Analysis name (used for file naming)
        r_code: Main R analysis code body
        data_file: Data file path
        required_packages: List of required R packages
        output_dir: Output directory

    Returns:
        str: Path to the generated .R file
    """
    if output_dir is None:
        output_dir = os.environ.get('OUTPUT_DIR', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    packages_str = ', '.join([f'"{pkg}"' for pkg in required_packages])

    r_script = f'''# ============================================================
# {analysis_name}
# Generated at: {timestamp}
# Data file: {data_file}
# ============================================================

# 1. Environment Setup - Automatically install missing packages
required_packages <- c({packages_str})
for (pkg in required_packages) {{
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }}
}}

# 2. Set working directory and output path
output_dir <- "{output_dir}"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 3. Read data
data <- read.csv("{data_file}")  # Or use readxl::read_excel()
cat("Data dimensions:", nrow(data), "rows x", ncol(data), "columns\\n")

# 4. Analysis code
{r_code}

# 5. Completion message
cat("\\nAnalysis complete! Results saved to:", output_dir, "\\n")
'''

    # Save R script
    script_path = os.path.join(output_dir, f'{analysis_name}.R')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(r_script)

    return script_path
```

### Common R Packages (Reference)

| Analysis Type | R Package | Purpose | Python Alternative |
|---------------|-----------|---------|-------------------|
| SEM | lavaan | Structural equation modeling | semopy (basic) |
| HLM | lme4, nlme | Mixed effects models | statsmodels (2-level) |
| Bayesian | brms, rstanarm | Bayesian regression | PyMC |
| Meta-analysis | metafor, meta | Meta-analysis | pymare (basic) |
| IRT | mirt, ltm | Item response theory | girth (limited) |
| Survival | survival, survminer | Survival analysis | lifelines |
| Visualization | ggplot2 | Advanced plotting | matplotlib/seaborn |

### SEM with lavaan (R Code Template)

```python
# Use when complex SEM is needed (multi-group comparison, measurement invariance, etc.)
r_code = '''
library(lavaan)

# Define model
model <- '
  # Measurement model
  latent1 =~ x1 + x2 + x3
  latent2 =~ y1 + y2 + y3

  # Structural model
  latent2 ~ latent1
'

# Fit model
fit <- sem(model, data = data)

# Results summary
summary(fit, fit.measures = TRUE, standardized = TRUE)

# Fit indices
fit_indices <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
print(round(fit_indices, 3))

# Save results
sink(file.path(output_dir, "SEM_results.txt"))
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("SEM results saved to SEM_results.txt\\n")
'''

script_path = generate_r_script(
    analysis_name="SEM_lavaan_analysis",
    r_code=r_code,
    data_file="data.csv",
    required_packages=["lavaan", "tidyverse"]
)
print(f"R script generated: {script_path}")
```

### HLM with lme4 (R Code Template)

```python
# Use when 3+ level HLM or complex random effects are needed
r_code = '''
library(lme4)
library(lmerTest)  # Provides p-values

# Null model (compute ICC)
model0 <- lmer(y ~ 1 + (1|group), data = data)
summary(model0)

# Compute ICC
var_comp <- as.data.frame(VarCorr(model0))
icc <- var_comp$vcov[1] / sum(var_comp$vcov)
cat("ICC =", round(icc, 3), "\\n")

# Random intercept model
model1 <- lmer(y ~ x1 + x2 + (1|group), data = data)
summary(model1)

# Random intercept and slope model
model2 <- lmer(y ~ x1 + x2 + (1 + x1|group), data = data)
summary(model2)

# Model comparison
anova(model1, model2)

# Save results
sink(file.path(output_dir, "HLM_results.txt"))
cat("=== Null Model ===\\n")
summary(model0)
cat("\\nICC =", round(icc, 3), "\\n")
cat("\\n=== Random Intercept Model ===\\n")
summary(model1)
cat("\\n=== Random Slope Model ===\\n")
summary(model2)
cat("\\n=== Model Comparison ===\\n")
print(anova(model1, model2))
sink()

cat("HLM results saved to HLM_results.txt\\n")
'''

script_path = generate_r_script(
    analysis_name="HLM_lme4_analysis",
    r_code=r_code,
    data_file="data.csv",
    required_packages=["lme4", "lmerTest", "tidyverse"]
)
```

### Meta-Analysis with metafor (R Code Template)

```python
# Use for complex meta-analysis (moderator effects, network meta-analysis, etc.)
r_code = '''
library(metafor)

# Assume data contains: yi (effect size), vi (variance), moderator (moderator variable)

# Random effects model
res <- rma(yi = yi, vi = vi, data = data, method = "REML")
summary(res)

# Heterogeneity test
cat("\\n=== Heterogeneity Test ===\\n")
cat("Q =", round(res$QE, 2), ", df =", res$k - 1, ", p =", format.pval(res$QEp), "\\n")
cat("I-squared =", round(res$I2, 1), "%\\n")
cat("tau-squared =", round(res$tau2, 4), "\\n")

# Publication bias test
cat("\\n=== Publication Bias Test ===\\n")
regtest(res)

# Forest plot
png(file.path(output_dir, "forest_plot.png"), width = 800, height = 600)
forest(res, slab = paste("Study", 1:nrow(data)))
dev.off()

# Funnel plot
png(file.path(output_dir, "funnel_plot.png"), width = 600, height = 600)
funnel(res)
dev.off()

# Moderator analysis (if moderator variable exists)
if ("moderator" %in% names(data)) {
  res_mod <- rma(yi = yi, vi = vi, mods = ~ moderator, data = data)
  summary(res_mod)
}

# Save results
sink(file.path(output_dir, "meta_analysis_results.txt"))
summary(res)
cat("\\n=== Publication Bias Test ===\\n")
regtest(res)
sink()

cat("Meta-analysis results saved\\n")
'''

script_path = generate_r_script(
    analysis_name="meta_analysis_metafor",
    r_code=r_code,
    data_file="meta_data.csv",
    required_packages=["metafor", "tidyverse"]
)
```

### IRT with mirt (R Code Template)

```python
# Use for IRT analysis (2PL, 3PL, GRM, etc.)
r_code = '''
library(mirt)

# Assume data contains only the response matrix (0/1 or polytomous)
items <- data[, grep("^item|^q", names(data))]  # Select item columns

# 2PL model
mod_2pl <- mirt(items, 1, itemtype = "2PL", verbose = FALSE)

# Model fit
M2(mod_2pl)

# Item parameters
cat("=== Item Parameters ===\\n")
coef(mod_2pl, simplify = TRUE, IRTpars = TRUE)$items

# Model fit indices
cat("\\n=== Model Fit ===\\n")
M2(mod_2pl)

# Item characteristic curves
png(file.path(output_dir, "ICC_plots.png"), width = 1000, height = 800)
plot(mod_2pl, type = "trace", facet_items = TRUE)
dev.off()

# Test information function
png(file.path(output_dir, "test_information.png"), width = 800, height = 600)
plot(mod_2pl, type = "info")
dev.off()

# Estimate ability scores
theta <- fscores(mod_2pl, method = "MAP")
data$theta <- theta[,1]

# Save results
sink(file.path(output_dir, "IRT_results.txt"))
cat("=== 2PL IRT Analysis Results ===\\n\\n")
cat("=== Item Parameters ===\\n")
print(coef(mod_2pl, simplify = TRUE, IRTpars = TRUE)$items)
cat("\\n=== Model Fit ===\\n")
print(M2(mod_2pl))
sink()

write.csv(data, file.path(output_dir, "data_with_theta.csv"), row.names = FALSE)
cat("IRT analysis complete, ability estimates added to data\\n")
'''

script_path = generate_r_script(
    analysis_name="IRT_mirt_analysis",
    r_code=r_code,
    data_file="item_responses.csv",
    required_packages=["mirt", "tidyverse"]
)
```

### RI-CLPM (Random Intercept Cross-Lagged Panel Model)

```python
# RI-CLPM is a typical complex model that requires R lavaan
r_code = '''
library(lavaan)

# RI-CLPM model syntax (3-wave data example)
ri_clpm_model <- '
  # Random intercepts (trait-like component)
  RI_X =~ 1*X1 + 1*X2 + 1*X3
  RI_Y =~ 1*Y1 + 1*Y2 + 1*Y3

  # Structured residuals (state-like component)
  WX1 =~ 1*X1
  WX2 =~ 1*X2
  WX3 =~ 1*X3
  WY1 =~ 1*Y1
  WY2 =~ 1*Y2
  WY3 =~ 1*Y3

  # Autoregressive paths
  WX2 ~ a*WX1
  WX3 ~ a*WX2
  WY2 ~ b*WY1
  WY3 ~ b*WY2

  # Cross-lagged paths
  WY2 ~ c*WX1
  WY3 ~ c*WX2
  WX2 ~ d*WY1
  WX3 ~ d*WY2

  # Covariances
  WX1 ~~ WY1
  WX2 ~~ WY2
  WX3 ~~ WY3
  RI_X ~~ RI_Y

  # Residual variances constrained to be equal
  X1 ~~ e*X1
  X2 ~~ e*X2
  X3 ~~ e*X3
  Y1 ~~ f*Y1
  Y2 ~~ f*Y2
  Y3 ~~ f*Y3
'

# Fit model
fit <- sem(ri_clpm_model, data = data, missing = "FIML")

# Results
summary(fit, fit.measures = TRUE, standardized = TRUE)

# Save
sink(file.path(output_dir, "RI_CLPM_results.txt"))
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("RI-CLPM results saved\\n")
'''

script_path = generate_r_script(
    analysis_name="RI_CLPM_analysis",
    r_code=r_code,
    data_file="panel_data.csv",
    required_packages=["lavaan"]
)
```

### User Instructions Template (For R Code Output)

When an R code file is generated, display the following to the user:

```python
def show_r_code_instructions(script_path: str, analysis_name: str, packages: list):
    """Generate user instruction message"""
    packages_install = ', '.join([f'"{p}"' for p in packages])

    instructions = f'''
## Warning: This Analysis Requires an R Environment

The requested analysis ({analysis_name}) exceeds the capabilities of Python libraries. An R script has been generated for you.

### How to Run

**Option 1: RStudio (Recommended)**
1. Open RStudio
2. Open file: `{script_path}`
3. Click "Source" or press Ctrl+Shift+Enter to run all

**Option 2: Command Line**
```bash
Rscript "{script_path}"
```

### First-time Setup: Install Required R Packages
```r
install.packages(c({packages_install}))
```

### Generated Files
- `{os.path.basename(script_path)}` - R analysis script
- After running, result files will be generated in the same directory
'''
    return instructions
```
