# Claude Statistical Analysis Skill

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

A Claude Code skill that transforms Claude into a professional statistical consultant. Instead of blindly running whatever method the user requests, it **diagnoses first, then analyzes** — checking data quality, validating assumptions, and automatically selecting the appropriate statistical method.

### What It Does

When you upload a dataset and describe your analysis needs, Claude will:

1. **Data Profile** (automatic) — Sample size, variable types, missing patterns, distributions, outliers
2. **Assumption Checking** — Normality, homogeneity of variance, multicollinearity; auto-switches methods when assumptions fail
3. **Execute Analysis** — Runs the appropriate statistical method with proper controls
4. **Output Triple** — Every analysis produces three deliverables:
   - APA 7th edition table (Excel + Markdown)
   - Publication-quality figure (300dpi PNG)
   - Results paragraph ready for manuscript (English/Chinese)

### Supported Methods

| Complexity | Methods | Workflow |
|-----------|---------|----------|
| Simple | Descriptive stats, t-test, chi-square, correlation, reliability | Fast path (0 confirmations) |
| Medium | Regression, ANOVA, moderation, mediation, ROC/AUC, survival | Light path (1 confirmation) |
| Complex | SEM/CFA, HLM, IRT, meta-analysis, RI-CLPM | Full path (3-4 confirmations) |
| Planning | Power analysis / sample size calculation | Dedicated path (no data needed) |

### Installation

#### As a Claude Code Skill

```bash
# Copy to your Claude skills directory
cp -r . ~/.claude/skills/statistical-analysis/
```

Claude Code will automatically detect and activate this skill when you:
- Upload a data file (.xlsx, .csv, .sav)
- Say "help me analyze" / "run statistical analysis"
- Mention a specific method (t-test, regression, SEM, etc.)

#### R Docker Environment (Optional)

For advanced methods that require R (SEM path diagrams, HLM, IRT):

```bash
cd docker/
chmod +x r-stat.sh
./r-stat.sh build    # Build the Docker image (~2GB)
./r-stat.sh test     # Verify installation
```

Pre-installed R packages: lavaan, lme4, metafor, mirt, psych, tidyverse, semPlot, effectsize, and 30+ more.

### Project Structure

```
.
├── SKILL.md                    # Core skill definition (Claude reads this)
├── references/
│   ├── methods-index.md        # Method selection decision trees
│   ├── code-patterns.md        # Python/R code templates
│   ├── table-formats.md        # APA table format specifications
│   └── full-workflow.md        # Full path detailed workflow
├── docker/
│   ├── Dockerfile              # R environment (rocker/tidyverse + 40 packages)
│   ├── r-stat.sh               # Convenience script for Docker operations
│   ├── README.md               # Docker setup guide
│   └── examples/               # SEM, HLM, meta-analysis R examples
└── assets/
    └── report-template.md      # Analysis report template
```

### Key Design Decisions

#### "Diagnose Before Analyze"

The v3 approach was "user says do X, we do X." The v4 approach is "check the data and assumptions first, then decide what to do." This prevents common mistakes like:

- Running parametric tests on non-normal data
- Missing multicollinearity in regression
- Using wrong effect size measures
- Ignoring missing data patterns

#### Automatic Method Switching

When assumptions fail, the skill automatically switches to the appropriate alternative and informs the user:

```
> Note: Variable X failed the normality test (Shapiro-Wilk p = .003),
> automatically switched to Mann-Whitney U test (non-parametric alternative).
```

#### Output Triple Enforcement

Every analysis **must** produce all three outputs (table + figure + paragraph). A built-in `check_output_triplet()` mechanism ensures nothing is skipped.

### Examples

**User**: "I have survey data, please analyze the relationship between gaming addiction and mental health, controlling for demographics."

**Claude will**:
1. Run data profile (N, distributions, missing patterns)
2. Check: normality, VIF, outliers
3. Recommend: hierarchical regression (Step 1: demographics, Step 2: gaming addiction)
4. Execute with HC3 robust standard errors (if heteroscedasticity detected)
5. Output: regression table + coefficient forest plot + APA results paragraph

### Requirements

- **Claude Code** with skills support
- **Python**: pandas, numpy, scipy, statsmodels, matplotlib, seaborn (installed automatically)
- **R Docker** (optional): Docker Desktop for advanced methods

---

<a id="中文"></a>

## 中文

一个 Claude Code 技能，将 Claude 打造为专业的统计分析顾问。它不会盲目执行用户请求的方法，而是**先诊断，再分析** — 检查数据质量、验证统计假设、自动选择合适的统计方法。

### 功能概述

上传数据集并描述分析需求后，Claude 会自动完成以下流程：

1. **数据画像**（自动执行）— 样本量、变量类型、缺失模式、分布特征、异常值检测
2. **假设检验** — 正态性、方差齐性、多重共线性检查；假设不满足时自动切换方法
3. **执行分析** — 运行合适的统计方法，包含必要的控制变量
4. **三件套输出** — 每次分析产出三项交付物：
   - APA 第7版格式表格（Excel + Markdown）
   - 出版级统计图表（300dpi PNG）
   - 可直接写入论文的结果段落（支持中英文）

### 支持的统计方法

| 复杂度 | 方法 | 工作流 |
|-------|------|--------|
| 简单 | 描述统计、t检验、卡方检验、相关分析、信效度分析 | 快速路径（0次确认） |
| 中等 | 回归分析、方差分析、调节效应、中介效应、ROC/AUC、生存分析 | 轻量路径（1次确认） |
| 复杂 | SEM/CFA、HLM、IRT、meta分析、RI-CLPM | 完整路径（3-4次确认） |
| 规划 | 统计功效分析 / 样本量计算 | 专用路径（无需数据） |

### 安装方法

#### 作为 Claude Code 技能安装

```bash
# 复制到 Claude 技能目录
cp -r . ~/.claude/skills/statistical-analysis/
```

Claude Code 会在以下场景自动激活此技能：
- 上传数据文件（.xlsx、.csv、.sav）
- 说"帮我分析"/"跑一下数据"/"统计分析"
- 提到具体统计方法（t检验、回归、SEM 等）

#### R Docker 环境（可选）

需要 R 的高级方法（SEM 路径图、HLM、IRT）：

```bash
cd docker/
chmod +x r-stat.sh
./r-stat.sh build    # 构建 Docker 镜像（约2GB）
./r-stat.sh test     # 验证安装
```

预装 R 包：lavaan、lme4、metafor、mirt、psych、tidyverse、semPlot、effectsize 等 30+ 个。

### 项目结构

```
.
├── SKILL.md                    # 核心技能定义文件（Claude 读取此文件）
├── references/
│   ├── methods-index.md        # 方法选择决策树
│   ├── code-patterns.md        # Python/R 代码模板
│   ├── table-formats.md        # APA 表格格式规范
│   └── full-workflow.md        # 完整路径详细工作流
├── docker/
│   ├── Dockerfile              # R 环境（rocker/tidyverse + 40个包）
│   ├── r-stat.sh               # Docker 操作便捷脚本
│   ├── README.md               # Docker 配置指南
│   └── examples/               # SEM、HLM、meta分析 R 示例
└── assets/
    └── report-template.md      # 分析报告模板
```

### 核心设计理念

#### "诊断先于分析"

v3 的做法是"用户说做什么就做什么"，v4 的做法是"先检查数据和假设，再决定怎么做"。这可以避免常见错误：

- 对非正态数据使用参数检验
- 回归分析中遗漏多重共线性
- 使用错误的效应量指标
- 忽视缺失数据模式

#### 自动方法切换

当假设不满足时，技能会自动切换到合适的替代方法并通知用户：

```
> 注意: 变量X未通过正态性检验 (Shapiro-Wilk p = .003),
> 已自动切换为 Mann-Whitney U 检验（非参数替代方法）。
```

#### 三件套输出强制机制

每次分析**必须**产出全部三项输出（表格 + 图表 + 段落）。内置的 `check_output_triplet()` 机制确保不会遗漏。

### 使用示例

**用户**："我有问卷调查数据，请分析游戏成瘾与心理健康的关系，需要控制人口学变量。"

**Claude 会自动**：
1. 运行数据画像（样本量、分布特征、缺失模式）
2. 检查：正态性、VIF、异常值
3. 推荐：层次回归（第一步：人口学变量，第二步：游戏成瘾）
4. 检测到异方差时自动使用 HC3 稳健标准误
5. 输出：回归系数表 + 系数森林图 + APA 格式结果段落

### 环境要求

- **Claude Code**（需支持 skills 功能）
- **Python**：pandas、numpy、scipy、statsmodels、matplotlib、seaborn（自动安装）
- **R Docker**（可选）：需安装 Docker Desktop

---

## License

MIT License - see [LICENSE](LICENSE)

## Changelog

- **v4.1** (2025-02-09): Mandatory output triple — figure generation embedded in all workflow paths
- **v4.0** (2025-02-09): "Diagnose before analyze" — data profiling, assumption checking, APA paragraph generation, medical research methods
- **v3.0** (2025-02-02): Three-tier path system, R Docker environment, APA table specifications
