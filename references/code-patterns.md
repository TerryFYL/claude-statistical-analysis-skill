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
        print(f"✅ Using Chinese font: {selected_font}")
    else:
        print("⚠️ No Chinese font found, text may display as boxes")
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
print(f"Bartlett's χ²: {chi_square:.2f}, p = {p_value:.3f}")

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
print(f"I²: {result['I2']:.1f}%")
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

## R Language Support (三级执行策略)

> **执行优先级**: Python → Docker R (自动) → 输出 .R 文件 (手动)

### 策略说明

```
当需要 R 执行时：
1. 检测 Docker R 环境是否可用
2. 如果可用 → 在 Docker 容器中执行 R 代码
3. 如果不可用 → 生成 .R 文件供用户在 RStudio 运行
```

---

### Docker R 执行 (优先方案)

#### Docker 环境构建 (首次配置)

**Dockerfile** (`docker/Dockerfile`):
```dockerfile
FROM rocker/tidyverse:4.3.2

# 中文支持
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 系统依赖
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*

# R 包安装
RUN Rscript -e 'install.packages(c( \
    "lavaan", "semPlot", "lme4", "lmerTest", \
    "metafor", "mirt", "brms", "survival", "survminer", \
    "psych", "car", "ggpubr", "corrplot", "openxlsx" \
), repos="https://cloud.r-project.org")'

WORKDIR /workspace
CMD ["R"]
```

**构建命令**:
```bash
cd docker
docker build -t r-stats-env .
```

#### Docker R 执行函数

```python
import subprocess
import shutil
import tempfile
import os

def check_docker_r_available():
    """检测 Docker R 环境是否可用"""
    if not shutil.which('docker'):
        return False, "Docker 未安装"

    try:
        result = subprocess.run(['docker', 'info'],
                                capture_output=True, timeout=5)
        if result.returncode != 0:
            return False, "Docker 未运行"
    except:
        return False, "Docker 连接失败"

    try:
        result = subprocess.run(
            ['docker', 'images', '-q', 'r-stats-env'],
            capture_output=True, text=True, timeout=10
        )
        if not result.stdout.strip():
            return False, "r-stats-env 镜像未构建"
    except:
        return False, "镜像检查失败"

    return True, "Docker R 环境可用"


def execute_r_in_docker(r_code: str, data_file: str = None,
                        output_dir: str = None):
    """
    在 Docker 容器中执行 R 代码

    Args:
        r_code: R 代码字符串
        data_file: 数据文件路径
        output_dir: 输出目录

    Returns:
        dict: {'success': bool, 'output': str, 'files': list}
    """
    if output_dir is None:
        output_dir = os.environ.get('OUTPUT_DIR', './outputs')

    os.makedirs(output_dir, exist_ok=True)
    workspace_dir = os.path.dirname(os.path.abspath(data_file)) if data_file else output_dir

    # 写入临时 R 脚本
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
        return {'success': False, 'output': '', 'errors': '执行超时 (300s)'}
    except Exception as e:
        return {'success': False, 'output': '', 'errors': str(e)}
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
```

#### Docker R 使用示例

```python
# 示例：在 Docker 中执行 lavaan SEM
r_code = '''
library(lavaan)

# 读取数据 (容器内路径)
data <- read.csv("/workspace/data.csv")

# SEM 模型
model <- '
  latent1 =~ x1 + x2 + x3
  latent2 =~ y1 + y2 + y3
  latent2 ~ latent1
'

fit <- sem(model, data = data)

# 输出结果
sink("/output/sem_results.txt")
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("SEM 分析完成！结果保存到 /output/sem_results.txt\\n")
'''

# 检测并执行
docker_ok, msg = check_docker_r_available()
if docker_ok:
    result = execute_r_in_docker(r_code, data_file="./data.csv")
    if result['success']:
        print("✅ Docker R 执行成功")
        print(result['output'])
    else:
        print(f"❌ 执行失败: {result['errors']}")
else:
    print(f"⚠️ Docker 不可用 ({msg})，降级到输出 .R 文件")
    # 调用 generate_r_script() ...
```

---

### 输出 .R 文件 (降级方案)

当 Docker R 不可用时，生成独立的 .R 文件供用户在本地 RStudio 运行。

#### R 代码文件生成函数

```python
import os
from datetime import datetime

def generate_r_script(analysis_name: str, r_code: str, data_file: str,
                      required_packages: list, output_dir: str = None):
    """
    生成独立的 R 脚本文件供用户本地运行

    Args:
        analysis_name: 分析名称 (用于文件命名)
        r_code: R 分析代码主体
        data_file: 数据文件路径
        required_packages: 所需 R 包列表
        output_dir: 输出目录

    Returns:
        str: 生成的 .R 文件路径
    """
    if output_dir is None:
        output_dir = os.environ.get('OUTPUT_DIR', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    packages_str = ', '.join([f'"{pkg}"' for pkg in required_packages])

    r_script = f'''# ============================================================
# {analysis_name}
# 生成时间: {timestamp}
# 数据文件: {data_file}
# ============================================================

# 1. 环境准备 - 自动安装缺失的包
required_packages <- c({packages_str})
for (pkg in required_packages) {{
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }}
}}

# 2. 设置工作目录和输出路径
output_dir <- "{output_dir}"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 3. 读取数据
data <- read.csv("{data_file}")  # 或使用 readxl::read_excel()
cat("数据维度:", nrow(data), "行 x", ncol(data), "列\\n")

# 4. 分析代码
{r_code}

# 5. 完成提示
cat("\\n分析完成！结果已保存到:", output_dir, "\\n")
'''

    # 保存 R 脚本
    script_path = os.path.join(output_dir, f'{analysis_name}.R')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(r_script)

    return script_path
```

### Common R Packages (参考)

| 分析类型 | R 包 | 用途 | Python 替代 |
|----------|------|------|-------------|
| SEM | lavaan | 结构方程模型 | semopy (基础) |
| HLM | lme4, nlme | 混合效应模型 | statsmodels (2层) |
| 贝叶斯 | brms, rstanarm | 贝叶斯回归 | PyMC |
| 元分析 | metafor, meta | Meta-analysis | pymare (基础) |
| IRT | mirt, ltm | 项目反应理论 | girth (有限) |
| 生存 | survival, survminer | 生存分析 | lifelines |
| 可视化 | ggplot2 | 高级绑图 | matplotlib/seaborn |

### SEM with lavaan (R 代码模板)

```python
# 当需要复杂 SEM (多组比较、测量不变性等) 时使用
r_code = '''
library(lavaan)

# 定义模型
model <- '
  # 测量模型
  latent1 =~ x1 + x2 + x3
  latent2 =~ y1 + y2 + y3

  # 结构模型
  latent2 ~ latent1
'

# 拟合模型
fit <- sem(model, data = data)

# 结果汇总
summary(fit, fit.measures = TRUE, standardized = TRUE)

# 拟合指数
fit_indices <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
print(round(fit_indices, 3))

# 保存结果
sink(file.path(output_dir, "SEM_results.txt"))
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("SEM 结果已保存到 SEM_results.txt\\n")
'''

script_path = generate_r_script(
    analysis_name="SEM_lavaan分析",
    r_code=r_code,
    data_file="data.csv",
    required_packages=["lavaan", "tidyverse"]
)
print(f"R 脚本已生成: {script_path}")
```

### HLM with lme4 (R 代码模板)

```python
# 当需要 3+ 水平 HLM 或复杂随机效应时使用
r_code = '''
library(lme4)
library(lmerTest)  # 提供 p 值

# 零模型 (计算 ICC)
model0 <- lmer(y ~ 1 + (1|group), data = data)
summary(model0)

# 计算 ICC
var_comp <- as.data.frame(VarCorr(model0))
icc <- var_comp$vcov[1] / sum(var_comp$vcov)
cat("ICC =", round(icc, 3), "\\n")

# 随机截距模型
model1 <- lmer(y ~ x1 + x2 + (1|group), data = data)
summary(model1)

# 随机截距和斜率模型
model2 <- lmer(y ~ x1 + x2 + (1 + x1|group), data = data)
summary(model2)

# 模型比较
anova(model1, model2)

# 保存结果
sink(file.path(output_dir, "HLM_results.txt"))
cat("=== 零模型 ===\\n")
summary(model0)
cat("\\nICC =", round(icc, 3), "\\n")
cat("\\n=== 随机截距模型 ===\\n")
summary(model1)
cat("\\n=== 随机斜率模型 ===\\n")
summary(model2)
cat("\\n=== 模型比较 ===\\n")
print(anova(model1, model2))
sink()

cat("HLM 结果已保存到 HLM_results.txt\\n")
'''

script_path = generate_r_script(
    analysis_name="HLM_lme4分析",
    r_code=r_code,
    data_file="data.csv",
    required_packages=["lme4", "lmerTest", "tidyverse"]
)
```

### Meta-Analysis with metafor (R 代码模板)

```python
# 复杂元分析 (调节效应、网络元分析等) 时使用
r_code = '''
library(metafor)

# 假设数据包含: yi (效应量), vi (方差), moderator (调节变量)

# 随机效应模型
res <- rma(yi = yi, vi = vi, data = data, method = "REML")
summary(res)

# 异质性检验
cat("\\n=== 异质性检验 ===\\n")
cat("Q =", round(res$QE, 2), ", df =", res$k - 1, ", p =", format.pval(res$QEp), "\\n")
cat("I² =", round(res$I2, 1), "%\\n")
cat("τ² =", round(res$tau2, 4), "\\n")

# 发表偏倚检验
cat("\\n=== 发表偏倚检验 ===\\n")
regtest(res)

# 森林图
png(file.path(output_dir, "forest_plot.png"), width = 800, height = 600)
forest(res, slab = paste("Study", 1:nrow(data)))
dev.off()

# 漏斗图
png(file.path(output_dir, "funnel_plot.png"), width = 600, height = 600)
funnel(res)
dev.off()

# 调节效应分析 (如果有调节变量)
if ("moderator" %in% names(data)) {
  res_mod <- rma(yi = yi, vi = vi, mods = ~ moderator, data = data)
  summary(res_mod)
}

# 保存结果
sink(file.path(output_dir, "meta_analysis_results.txt"))
summary(res)
cat("\\n=== 发表偏倚检验 ===\\n")
regtest(res)
sink()

cat("元分析结果已保存\\n")
'''

script_path = generate_r_script(
    analysis_name="元分析_metafor",
    r_code=r_code,
    data_file="meta_data.csv",
    required_packages=["metafor", "tidyverse"]
)
```

### IRT with mirt (R 代码模板)

```python
# IRT 分析 (2PL, 3PL, GRM 等) 时使用
r_code = '''
library(mirt)

# 假设数据只包含作答矩阵 (0/1 或多类别)
items <- data[, grep("^item|^q", names(data))]  # 选择题目列

# 2PL 模型
mod_2pl <- mirt(items, 1, itemtype = "2PL", verbose = FALSE)

# 模型拟合
M2(mod_2pl)

# 题目参数
cat("=== 题目参数 ===\\n")
coef(mod_2pl, simplify = TRUE, IRTpars = TRUE)$items

# 模型拟合指数
cat("\\n=== 模型拟合 ===\\n")
M2(mod_2pl)

# 项目特征曲线
png(file.path(output_dir, "ICC_plots.png"), width = 1000, height = 800)
plot(mod_2pl, type = "trace", facet_items = TRUE)
dev.off()

# 测验信息函数
png(file.path(output_dir, "test_information.png"), width = 800, height = 600)
plot(mod_2pl, type = "info")
dev.off()

# 估计能力值
theta <- fscores(mod_2pl, method = "MAP")
data$theta <- theta[,1]

# 保存结果
sink(file.path(output_dir, "IRT_results.txt"))
cat("=== 2PL IRT 分析结果 ===\\n\\n")
cat("=== 题目参数 ===\\n")
print(coef(mod_2pl, simplify = TRUE, IRTpars = TRUE)$items)
cat("\\n=== 模型拟合 ===\\n")
print(M2(mod_2pl))
sink()

write.csv(data, file.path(output_dir, "data_with_theta.csv"), row.names = FALSE)
cat("IRT 分析完成，能力估计值已添加到数据\\n")
'''

script_path = generate_r_script(
    analysis_name="IRT_mirt分析",
    r_code=r_code,
    data_file="item_responses.csv",
    required_packages=["mirt", "tidyverse"]
)
```

### RI-CLPM (随机截距交叉滞后面板模型)

```python
# RI-CLPM 是典型的需要 R lavaan 的复杂模型
r_code = '''
library(lavaan)

# RI-CLPM 模型语法 (3 波数据示例)
ri_clpm_model <- '
  # 随机截距 (trait-like 成分)
  RI_X =~ 1*X1 + 1*X2 + 1*X3
  RI_Y =~ 1*Y1 + 1*Y2 + 1*Y3

  # 结构化残差 (state-like 成分)
  WX1 =~ 1*X1
  WX2 =~ 1*X2
  WX3 =~ 1*X3
  WY1 =~ 1*Y1
  WY2 =~ 1*Y2
  WY3 =~ 1*Y3

  # 自回归路径
  WX2 ~ a*WX1
  WX3 ~ a*WX2
  WY2 ~ b*WY1
  WY3 ~ b*WY2

  # 交叉滞后路径
  WY2 ~ c*WX1
  WY3 ~ c*WX2
  WX2 ~ d*WY1
  WX3 ~ d*WY2

  # 协方差
  WX1 ~~ WY1
  WX2 ~~ WY2
  WX3 ~~ WY3
  RI_X ~~ RI_Y

  # 残差方差约束为相等
  X1 ~~ e*X1
  X2 ~~ e*X2
  X3 ~~ e*X3
  Y1 ~~ f*Y1
  Y2 ~~ f*Y2
  Y3 ~~ f*Y3
'

# 拟合模型
fit <- sem(ri_clpm_model, data = data, missing = "FIML")

# 结果
summary(fit, fit.measures = TRUE, standardized = TRUE)

# 保存
sink(file.path(output_dir, "RI_CLPM_results.txt"))
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("RI-CLPM 结果已保存\\n")
'''

script_path = generate_r_script(
    analysis_name="RI_CLPM分析",
    r_code=r_code,
    data_file="panel_data.csv",
    required_packages=["lavaan"]
)
```

### 用户提示模板 (配合 R 代码输出)

当生成 R 代码文件后，向用户展示：

```python
def show_r_code_instructions(script_path: str, analysis_name: str, packages: list):
    """生成用户指引信息"""
    packages_install = ', '.join([f'"{p}"' for p in packages])

    instructions = f'''
## ⚠️ 此分析需要 R 环境

当前请求的分析（{analysis_name}）超出 Python 库的覆盖能力，已为您生成 R 代码。

### 运行方式

**方式一：RStudio (推荐)**
1. 打开 RStudio
2. 打开文件：`{script_path}`
3. 点击 "Source" 或 Ctrl+Shift+Enter 运行全部

**方式二：命令行**
```bash
Rscript "{script_path}"
```

### 首次运行需安装 R 包
```r
install.packages(c({packages_install}))
```

### 生成的文件
- `{os.path.basename(script_path)}` - R 分析脚本
- 运行后将生成结果文件到同一目录
'''
    return instructions
```
