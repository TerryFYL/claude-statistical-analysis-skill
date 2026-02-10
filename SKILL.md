---
name: statistical-analysis
description: >
  统计分析服务。触发条件: (1) 上传数据文件, (2) 说"统计分析"/"帮我分析"/"跑一下数据",
  (3) 提到具体统计方法(t检验/回归/SEM等)。

  **核心原则**: 像顶级统计顾问一样主动诊断，不只是执行用户说的方法，而是确保方法选择正确。
---

# Statistical Analysis Service v4

## 核心理念: 诊断先于分析

v3 是"用户说做什么就做什么"，v4 是"先诊断数据和假设，再决定怎么做"。

```
v3: 用户请求 → 选路径 → 执行
v4: 用户请求 → 数据画像 → 假设检查 → 智能选择方法 → 执行 → 三件套输出(表格+图表+段落)
```

---

## 流程总览

```
用户请求
    │
    ▼
┌─────────────────────────────────────┐
│  Step 0: 数据画像（所有路径必做）     │
│  • 样本量、变量类型、缺失模式        │
│  • 分布特征、异常值检测              │
│  • 30秒内完成，不需确认              │
└─────────────────────────────────────┘
    │
    ▼
判断复杂度 → 选择路径
    │
    ├── 快速路径 → 假设自检 → 执行 → 三件套输出
    ├── 轻量路径 → 假设自检 → 确认变量 → 执行 → 三件套输出
    └── 完整路径 → 四阶段（含假设检查）→ 三件套输出
```

---

## Step 0: 数据画像（Data Profile）

**所有分析之前必须执行**，输出格式：

```markdown
## 数据画像

| 指标 | 值 |
|------|-----|
| 样本量 | N = 3248 |
| 变量数 | 94 (连续: 60, 分类: 34) |
| 缺失率 | 整体 2.3%，最高: 变量X (15.2%) |
| 异常值 | 变量Y 有 12 个 (> 3SD) |

### 关键变量分布
| 变量 | M (SD) | 偏度 | 峰度 | 正态性 |
|------|--------|------|------|--------|
| DV_score | 3.45 (1.23) | 0.34 | -0.12 | ✅ 通过 |
| IV_score | 2.89 (0.98) | 1.45 | 3.21 | ❌ 右偏 |

### 数据提醒
- ⚠️ IV_score 呈右偏分布，参数检验需谨慎
- ⚠️ 变量X 缺失 15%，建议检查 MCAR/MAR
- ✅ 样本量充足，支持所有常规分析
```

**执行代码**（自动运行，不展示给用户）：

```python
import pandas as pd
import numpy as np
from scipy import stats

def data_profile(df, target_vars=None):
    """生成数据画像，target_vars 为用户提到的关键变量"""
    vars_to_check = target_vars or df.select_dtypes(include=[np.number]).columns[:10]

    profile = {}
    for var in vars_to_check:
        col = df[var].dropna()
        n = len(col)
        # 正态性检验: n<50 用 Shapiro-Wilk, n>=50 用偏度+峰度判断
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

## 复杂度判断与路径选择

| 复杂度 | 分析类型 | 路径 | 确认次数 |
|--------|----------|------|----------|
| **简单** | 描述统计、t检验、卡方、相关、信效度 | 快速 | 0 |
| **中等** | 回归、ANOVA、调节、中介、ROC/AUC、生存分析 | 轻量 | 1 |
| **复杂** | SEM/CFA、HLM、IRT、元分析、RI-CLPM、倾向性得分匹配 | 完整 | 3-4 |
| **规划** | 样本量计算/Power Analysis（无数据，仅参数） | 专用 | 1 |

---

## 快速路径（简单分析）

**适用**: 描述统计、t检验、卡方、相关、信效度

**流程**: 数据画像 → 假设自检 → 执行 → 三件套输出(表格+图表+段落)

### 假设自检（自动，嵌入执行过程）

```python
def check_and_run_ttest(df, group_var, value_var, group1, group2):
    """自动检查假设并选择合适的t检验"""
    g1 = df[df[group_var] == group1][value_var].dropna()
    g2 = df[df[group_var] == group2][value_var].dropna()

    # 1. 正态性检验
    n1, n2 = len(g1), len(g2)
    if min(n1, n2) < 50:
        _, p1 = stats.shapiro(g1)
        _, p2 = stats.shapiro(g2)
        normal = p1 > .05 and p2 > .05
    else:
        normal = abs(g1.skew()) < 2 and abs(g2.skew()) < 2

    if not normal:
        # 非参数替代
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        method = "Mann-Whitney U"
        effect = abs(stat - n1*n2/2) / (n1*n2)  # rank-biserial r
        return method, stat, p, effect

    # 2. 方差齐性检验
    _, p_levene = stats.levene(g1, g2)
    equal_var = p_levene > .05

    # 3. 选择 t 检验类型
    stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
    method = "独立样本 t 检验" if equal_var else "Welch's t 检验"

    # 4. Cohen's d
    pooled_sd = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    d = (g1.mean() - g2.mean()) / pooled_sd

    return method, stat, p, d
```

**关键行为**: 当假设不满足时，**自动切换方法并告知用户**：

```markdown
> 注意: 变量X 未通过正态性检验 (Shapiro-Wilk p = .003)，
> 已自动切换为 Mann-Whitney U 检验（非参数替代）。
```

---

## 轻量路径（中等分析）

**适用**: 回归、ANOVA、调节效应、中介效应、ROC/AUC、生存分析

**流程**: 数据画像 → 假设自检 → 确认变量 → 执行 → 三件套输出(表格+图表+段落)

### 确认变量（增强版）

```markdown
数据已读取: N = 3005

请确认变量角色:
| 角色 | 变量 | 类型 | 分布状态 |
|------|------|------|----------|
| 因变量(Y) | SDQ总分 | 连续 | ✅ 正态 |
| 自变量(X) | 游戏障碍评分 | 连续 | ⚠️ 右偏 (偏度=1.45) |
| 调节变量(M) | 独生子女 | 二分 | — |
| 控制变量 | 年龄、性别 | 连续/二分 | ✅ |

### 自动假设检查结果
- ✅ 样本量充足 (N=3005, 远超最低要求)
- ✅ 多重共线性: VIF 均 < 5
- ⚠️ 自变量右偏，建议考虑: (a) 对数变换 (b) 稳健标准误 (c) 保持原样
- 推荐: 使用稳健标准误 (HC3)，保持变量原始含义

确认后继续分析。
```

---

## 完整路径（复杂分析）

**适用**: SEM/CFA、HLM、IRT、元分析、RI-CLPM

**流程**: 四阶段，每阶段确认（详见 `references/full-workflow.md`）

```
阶段1: 数据画像 + 清洗方案 → ⏸️确认
阶段2: 执行清洗 + 假设检查 → ⏸️确认
阶段3: 分析方案 + 样本量充分性 → ⏸️确认
阶段4: 执行分析 → 三件套输出(表格+图表+段落)
```

---

## Power Analysis 路径（样本量计算）

**触发**: 用户说"样本量"/"统计检验力"/"power analysis"/"需要多少人"

**无需数据文件**，只需参数：

```markdown
## 样本量计算

请提供以下信息:
| 参数 | 你的设定 | 默认值 |
|------|----------|--------|
| 分析方法 | ? | — |
| 预期效应量 | ? | 中等 (d=0.5 / f=0.25 / r=.30) |
| 显著性水平 (α) | ? | .05 |
| 统计检验力 (1-β) | ? | .80 |
| 组数 | ? | 2 |
| 是否单侧 | ? | 双侧 |
```

**执行**:
```python
from scipy import stats
import numpy as np

def power_ttest(d, alpha=0.05, power=0.80, ratio=1, alternative='two-sided'):
    """t检验样本量计算 (每组)"""
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

# 常用方法的样本量速查
power_table = {
    't检验': {'小(d=0.2)': 394, '中(d=0.5)': 64, '大(d=0.8)': 26},
    'ANOVA(3组)': {'小(f=0.1)': 969, '中(f=0.25)': 159, '大(f=0.4)': 66},
    '相关': {'小(r=.1)': 783, '中(r=.3)': 85, '大(r=.5)': 29},
    '回归(3个IV)': {'小(f²=.02)': 550, '中(f²=.15)': 77, '大(f²=.35)': 36},
}
```

---

## APA 结果段落生成（所有路径的最终输出）

**每次分析完成后，除了表格和图表，必须生成可直接放入论文的结果段落。**

### 结果段落模板

**t检验**:
> An independent samples t-test revealed a significant difference in {DV} between {group1} (M = {m1}, SD = {sd1}) and {group2} (M = {m2}, SD = {sd2}), t({df}) = {t}, p {p_text}, Cohen's d = {d}. The effect size was {small/medium/large}.

**相关分析**:
> Pearson correlation analysis showed that {var1} was significantly {positively/negatively} correlated with {var2}, r({df}) = {r}, p {p_text}. The correlation coefficient indicated a {small/medium/large} effect.

**回归分析**:
> A hierarchical multiple regression was conducted. In Step 1, {controls} were entered, accounting for {R1²}% of variance in {DV}, F({df1}, {df2}) = {F1}, p {p1}. In Step 2, {predictor} was added, significantly improving model fit, ΔR² = {dr2}, ΔF({ddf1}, {ddf2}) = {dF}, p {dp}. {Predictor} was a significant predictor (β = {beta}, p {p_text}).

**中介效应**:
> A mediation analysis using 5000 bootstrap samples revealed a significant indirect effect of {X} on {Y} through {M} (indirect effect = {ab}, 95% CI [{ci_lo}, {ci_hi}]). The direct effect was {significant/non-significant} (c' = {c_prime}, p {p_text}), suggesting {full/partial} mediation.

**调节效应**:
> The interaction between {X} and {W} was significant (B = {b3}, SE = {se}, p {p_text}), indicating that {W} moderated the relationship between {X} and {Y}. Simple slope analysis showed that the effect of {X} on {Y} was significant at high levels of {W} (+1SD: B = {b_high}, p {p_high}) but not at low levels (-1SD: B = {b_low}, p {p_low}).

### 结果段落代码

```python
def format_p(p):
    """APA格式 p 值"""
    if p < .001: return "< .001"
    return f"= {p:.3f}"

def effect_size_label(d):
    """Cohen's d 效应量描述"""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

def write_ttest_result(group1_name, group2_name, dv_name,
                        m1, sd1, m2, sd2, t, df, p, d, method="Independent samples t-test"):
    """生成 t 检验结果段落"""
    sig = "significant" if p < .05 else "non-significant"
    return (
        f"An {method.lower()} revealed a {sig} difference in {dv_name} "
        f"between {group1_name} (M = {m1:.2f}, SD = {sd1:.2f}) and "
        f"{group2_name} (M = {m2:.2f}, SD = {sd2:.2f}), "
        f"t({df}) = {t:.2f}, p {format_p(p)}, Cohen's d = {d:.2f}. "
        f"The effect size was {effect_size_label(d)}."
    )
```

**输出语言**: 默认英文（论文通用）。如果用户说"中文"，则生成中文版本。

---

## 图表自动生成引擎（所有路径必须执行）

**核心规则**: 每次分析执行后，必须自动生成对应图表。图表不是可选附件，而是三件套的必要组成部分。

### 初始化（每次绘图前自动执行）

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def init_figure(figsize=(10, 6)):
    """统一图表初始化: 中文字体 + APA风格"""
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

### 分析类型 → 图表函数映射

| 分析类型 | 图表函数 | 输出文件名 |
|----------|----------|------------|
| 描述统计 | `plot_distribution()` | `fig_distribution.png` |
| 组间比较 (t检验) | `plot_group_comparison()` | `fig_group_comparison.png` |
| 相关分析 | `plot_correlation_heatmap()` | `fig_correlation_heatmap.png` |
| 回归分析 | `plot_regression_coefficients()` | `fig_regression_coef.png` |
| 层次回归 | `plot_hierarchical_regression()` | `fig_hierarchical_reg.png` |
| ANOVA | `plot_anova_boxplot()` | `fig_anova_boxplot.png` |
| 调节效应 | `plot_moderation_interaction()` | `fig_moderation.png` |
| 中介效应 | `plot_mediation_path()` | `fig_mediation_path.png` |
| ROC/AUC | `plot_roc_curve()` | `fig_roc_curve.png` |
| 生存分析 | `plot_km_curve()` | `fig_km_curve.png` |

### 图表代码模板

**组间比较柱状图** (t检验/卡方后使用):
```python
def plot_group_comparison(means, sds, group_labels, dv_labels, title, save_path,
                           p_values=None, figsize=(10, 6)):
    """带误差棒的分组柱状图"""
    fig, ax = init_figure(figsize)
    x = np.arange(len(dv_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, means[0], width, yerr=sds[0], capsize=4,
                   color='#4C72B0', alpha=0.85, label=group_labels[0])
    bars2 = ax.bar(x + width/2, means[1], width, yerr=sds[1], capsize=4,
                   color='#DD8452', alpha=0.85, label=group_labels[1])
    # 显著性星号
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

**相关热力图**:
```python
def plot_correlation_heatmap(corr_matrix, p_matrix, title, save_path, figsize=(12, 10)):
    """带显著性星号的相关热力图"""
    fig, ax = init_figure(figsize)
    # 生成星号标注
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

**回归系数森林图**:
```python
def plot_regression_coefficients(names, betas, ci_lower, ci_upper, title, save_path,
                                  figsize=(8, 6)):
    """标准化回归系数森林图 (含95%CI)"""
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
    ax.set_xlabel('标准化回归系数 (β)')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**层次回归双面板图**:
```python
def plot_hierarchical_regression(step_labels, r2_values, delta_r2, predictor_names,
                                  betas, title, save_path, figsize=(14, 6)):
    """左: ΔR²堆叠柱 | 右: β系数横向柱"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # 左图: ΔR² 堆叠
    bottom = 0
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for i, (label, dr2) in enumerate(zip(step_labels, delta_r2)):
        ax1.bar(0, dr2, bottom=bottom, color=colors[i % len(colors)],
                alpha=0.85, label=f'{label} (ΔR²={dr2:.3f})')
        bottom += dr2
    ax1.set_ylabel('R²')
    ax1.set_title('模型解释量', fontweight='bold')
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_xticks([])
    # 右图: β 系数
    y_pos = np.arange(len(predictor_names))
    colors_beta = ['#C44E52' if b > 0 else '#4C72B0' for b in betas]
    ax2.barh(y_pos, betas, color=colors_beta, alpha=0.7, height=0.6)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictor_names)
    ax2.set_xlabel('标准化系数 (β)')
    ax2.set_title('预测因子贡献', fontweight='bold')
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**调节效应交互图**:
```python
def plot_moderation_interaction(x_vals, y_low, y_high, x_label, y_label,
                                 mod_label, title, save_path, figsize=(8, 6)):
    """简单斜率图: 调节变量高/低水平下X-Y关系"""
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

**中介效应路径图**:
```python
def plot_mediation_path(x_name, m_name, y_name, a, b, c, c_prime, indirect,
                         ci_lo, ci_hi, title, save_path, figsize=(10, 6)):
    """中介路径图: X→M→Y + 直接效应"""
    fig, ax = init_figure(figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    # 框
    for (cx, cy, label) in [(1.5, 3, x_name), (5, 5, m_name), (8.5, 3, y_name)]:
        ax.add_patch(plt.Rectangle((cx-1, cy-0.4), 2, 0.8,
                     fill=True, facecolor='#E8E8E8', edgecolor='black', linewidth=1.5))
        ax.text(cx, cy, label, ha='center', va='center', fontsize=11, fontweight='bold')
    # 箭头 + 系数
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
    ax.text(5, 1.2, f'间接效应 = {indirect:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

**ANOVA 箱线图**:
```python
def plot_anova_boxplot(df, group_var, dv_vars, title, save_path, figsize=(14, 8)):
    """多因变量分组箱线图网格"""
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

## 三件套输出检查（每次分析结束时自动执行）

**规则**: 每完成一次分析，必须自检三件套是否齐全。缺少任何一项则自动补齐。

```
分析完成
    │
    ▼
检查三件套:
    ├── ✅ APA表格 (Excel/Markdown) → 已生成
    ├── ✅ 图表 (PNG, dpi=300) → 已生成
    └── ✅ 结果段落 → 已生成
           │
           └── 全部 ✅ → 输出完成
           └── 有 ❌ → 自动补齐缺失项
```

**自检代码**（内嵌在分析流程结尾）:
```python
def check_output_triplet(analysis_type, outputs):
    """检查三件套是否齐全"""
    required = ['table', 'figure', 'paragraph']
    missing = [r for r in required if r not in outputs or outputs[r] is None]
    if missing:
        print(f"⚠️ {analysis_type} 缺少: {', '.join(missing)}，正在自动补齐...")
    return missing
```

**行为约束**:
- 当 `check_output_triplet()` 返回非空列表时，**必须立即补齐**
- 不允许以"用户没要求图表"为由跳过
- 图表类型由"分析类型→图表函数映射"表自动决定

---

## 医学研究专用方法

### 信效度分析（快速路径）

**触发**: "信度"/"效度"/"Cronbach"/"量表验证"/"内部一致性"

```python
import pingouin as pg

# Cronbach's α
alpha = pg.cronbach_alpha(df[items])

# 如果 α < .70 → 建议检查各题项
if alpha[0] < .70:
    # 逐题删除后的 α
    for item in items:
        remaining = [i for i in items if i != item]
        a = pg.cronbach_alpha(df[remaining])
        print(f"删除 {item} 后 α = {a[0]:.3f}")
```

**输出**: α系数 + 各题项修正后总相关 (CITC) + 删除后α表

### ROC/AUC 分析（轻量路径）

**触发**: "ROC"/"AUC"/"诊断准确性"/"敏感度"/"特异度"/"截断值"

```python
from sklearn.metrics import roc_curve, auc, confusion_matrix

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 最佳截断值 (Youden's index)
youden = tpr - fpr
optimal_idx = np.argmax(youden)
optimal_threshold = thresholds[optimal_idx]

# 在最佳截断值下的指标
y_pred = (y_scores >= optimal_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
```

**输出**: ROC曲线图 + AUC + 最佳截断值 + 敏感度/特异度/PPV/NPV表

### 生存分析（轻量路径）

**触发**: "生存分析"/"Kaplan-Meier"/"Cox"/"风险比"/"HR"/"生存曲线"

代码模板见 `references/code-patterns.md` → Survival Analysis 部分

### 组内相关系数 ICC（快速路径）

**触发**: "ICC"/"评分者一致性"/"评分者信度"/"组内相关"

```python
import pingouin as pg

# ICC(3,1) - 最常用：双向混合、一致性、单个测量
icc = pg.intraclass_corr(data=df_long, targets='subject',
                          raters='rater', ratings='score')
# 报告 ICC(3,1) 行
```

---

## 执行策略

### 默认：Python（缺包就装）

```bash
pip3 install scipy statsmodels pingouin matplotlib seaborn scikit-learn --quiet --break-system-packages
```

**中文绘图**:
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
```

### 方法-库 速查表

| 方法 | 库 | 示例 |
|------|-----|------|
| 描述统计 | pandas | `df.describe()` |
| t检验 | scipy | `stats.ttest_ind(a, b)` |
| 卡方检验 | scipy | `stats.chi2_contingency(table)` |
| 相关分析 | pandas/scipy | `df.corr()` / `pearsonr()` |
| 回归 | statsmodels | `sm.OLS(y, X).fit()` |
| ANOVA | pingouin | `pg.anova(data, dv, between)` |
| 调节效应 | statsmodels | 交互项回归 |
| 中介效应 | 自定义 | Bootstrap (见 code-patterns) |
| 信效度 | pingouin | `pg.cronbach_alpha()` |
| ROC/AUC | sklearn | `roc_curve()` / `auc()` |
| 生存分析 | lifelines | `KaplanMeierFitter` / `CoxPHFitter` |
| ICC | pingouin | `pg.intraclass_corr()` |
| Power | scipy | 自定义函数 (见上文) |

### 仅当用户要求或必须时：R Docker

**触发条件**：
- 用户明确说"用R"/"用lavaan"/"用lme4"
- 分析方法 Python 无法实现（SEM路径图、HLM随机效应图）

```bash
# R Docker 位置
cd docker/

# 快速使用
./r-stat.sh build    # 首次构建
./r-stat.sh run x.R  # 执行脚本
```

**已安装R包**: lavaan, lme4, metafor, mirt, psych, tidyverse, semPlot, effectsize, parameters, performance

---

## 输出规范

### 必须输出的三件套

每次分析必须输出:
1. **APA 表格** (Excel + Markdown) — 格式见 `references/table-formats.md`
2. **图表** (PNG, dpi=300) — 类型见下表
3. **结果段落** (英文，可切中文) — 可直接放入论文

### 表格格式 (APA 7)

| 项目 | 规范 |
|------|------|
| 统计量 | 2位小数 |
| p值 | 3位小数，<.001 写 "<.001" |
| 显著性 | *p<.05, **p<.01, ***p<.001 |
| 效应量 | Cohen's d, η², R², f² |
| 置信区间 | 95% CI [lower, upper] |

### 图表默认输出

| 分析类型 | 默认图表 |
|----------|----------|
| 描述统计 | 分布图/箱线图 |
| 相关分析 | 热力图 |
| 回归 | 系数森林图 |
| 调节效应 | 简单斜率图 |
| 中介效应 | 路径图 |
| ROC | ROC曲线 (含AUC) |
| 生存分析 | K-M生存曲线 |
| SEM/CFA | 路径图 |
| HLM | 个体轨迹图 |
| 元分析 | 森林图 + 漏斗图 |

---

## 缺失数据处理策略

当缺失率 > 5% 时，自动提醒并建议处理策略:

| 缺失率 | 建议策略 | 说明 |
|--------|----------|------|
| < 5% | 列表删除 (listwise) | 影响小，简单处理 |
| 5-20% | 多重填补 (MICE) | 保留样本量，减少偏倚 |
| > 20% | 检查 MCAR → 决定策略 | 需要 Little's MCAR 检验 |

```python
# Little's MCAR 检验 (近似)
# 如果显著 → 数据不是完全随机缺失，需谨慎处理
from sklearn.impute import SimpleImputer
# 推荐: 使用多重填补并报告敏感性分析
```

---

## 资源文件

| 文件 | 用途 |
|------|------|
| `references/methods-index.md` | 方法选择决策树 (含医学专用方法) |
| `references/code-patterns.md` | 代码模板 (Python + R) |
| `references/full-workflow.md` | 完整路径详细流程 |
| `references/table-formats.md` | APA表格格式模板 |
| `docker/` | R Docker 执行环境 |
| `docker/examples/` | SEM/HLM/元分析/IRT 示例 |

---

## 更新日志

- **v4.1** (2026-02-09): 三件套强制输出
  - **修复**: 图表生成从"规范描述"升级为"流程内嵌" — 所有路径终点改为三件套输出
  - 新增 **图表自动生成引擎**: 8种分析类型的可执行绘图代码模板
  - 新增 **三件套输出检查机制**: 分析结束自检 表格+图表+段落，缺项自动补齐
  - 新增 **init_figure()**: 统一中文字体/DPI/APA风格初始化
  - 新增 **分析类型→图表函数映射表**: 10种分析对应的默认图表类型
  - 修复: 快速/轻量/完整三条路径流程描述均以"三件套输出"结尾
- **v4.0** (2026-02-09): 诊断先于分析
  - 新增 **Step 0 数据画像**: 所有分析前自动执行数据质量检查
  - 新增 **假设自检引擎**: 自动检查前提假设，不满足时自动切换方法
  - 新增 **APA结果段落生成**: 输出可直接放入论文的结果描述
  - 新增 **医学专用方法**: 信效度分析、ROC/AUC、生存分析路由、ICC、Power Analysis
  - 新增 **缺失数据策略**: 根据缺失率自动建议处理方法
  - 新增 **Power Analysis 专用路径**: 样本量计算独立流程
  - 升级 **复杂度判断表**: 加入信效度(简单)、ROC/生存分析(中等)
  - 升级 **轻量路径确认**: 展示假设检查结果和数据提醒
  - 核心理念从"按复杂度选路径"升级为"诊断先于分析"
- **v3.0** (2026-02-02): 三级路径系统
  - 快速/轻量/完整三级路径
  - R Docker 执行环境
  - APA 表格格式规范
