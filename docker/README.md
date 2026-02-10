# Statistical Analysis R Docker 环境

> 社会科学/心理学统计分析专用 R 环境，支持 SEM、HLM、元分析、IRT 等高级分析方法。

---

## 快速开始

### 1. 构建镜像

```bash
cd docker/
chmod +x r-stat.sh
./r-stat.sh build
```

### 2. 运行分析

```bash
# 运行 R 脚本
./r-stat.sh run analysis.R

# 交互式 R
./r-stat.sh shell

# 执行代码片段
./r-stat.sh eval 'library(lavaan); packageVersion("lavaan")'
```

### 3. 验证环境

```bash
./r-stat.sh test
```

---

## 包含的 R 包

### 结构方程模型 (SEM)

| 包名 | 版本 | 用途 |
|------|------|------|
| lavaan | 0.6+ | SEM/CFA 核心引擎 |
| semPlot | - | 路径图可视化 |
| semTools | - | SEM 辅助工具 |

**示例**:
```r
library(lavaan)

model <- '
  # 测量模型
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9

  # 结构模型
  speed ~ visual + textual
'

fit <- sem(model, data = HolzingerSwineford1939)
summary(fit, fit.measures = TRUE, standardized = TRUE)
```

### 多层线性模型 (HLM)

| 包名 | 用途 |
|------|------|
| lme4 | 混合效应模型核心 |
| lmerTest | 提供 p 值 |
| nlme | 传统混合模型 |

**示例**:
```r
library(lme4)
library(lmerTest)

# 随机截距模型
model <- lmer(score ~ treatment + (1|school), data = mydata)

# 随机斜率模型
model <- lmer(score ~ treatment + (1 + treatment|school), data = mydata)

summary(model)
```

### 元分析

| 包名 | 用途 |
|------|------|
| metafor | 元分析核心 |
| meta | 简化接口 |

**示例**:
```r
library(metafor)

# 效应量计算
dat <- escalc(measure = "SMD",
              m1i = mean_exp, sd1i = sd_exp, n1i = n_exp,
              m2i = mean_ctrl, sd2i = sd_ctrl, n2i = n_ctrl,
              data = mydata)

# 随机效应模型
res <- rma(yi, vi, data = dat, method = "REML")
summary(res)

# 森林图
forest(res)
```

### 项目反应理论 (IRT)

| 包名 | 用途 |
|------|------|
| mirt | IRT 分析核心 |
| ltm | 传统 IRT 模型 |

**示例**:
```r
library(mirt)

# 2PL 模型
fit <- mirt(response_matrix, 1, itemtype = "2PL")

# 项目参数
coef(fit, simplify = TRUE)

# 能力估计
theta <- fscores(fit)

# 项目特征曲线
plot(fit, type = "trace")
```

### 心理测量与因子分析

| 包名 | 用途 |
|------|------|
| psych | 心理测量工具包 |
| GPArotation | 因子旋转 |
| nFactors | 因子数量确定 |

### 效应量与报告

| 包名 | 用途 |
|------|------|
| effectsize | 效应量计算 |
| parameters | 参数提取 |
| performance | 模型评估 |
| report | 自动报告生成 |

### 中介与调节

| 包名 | 用途 |
|------|------|
| mediation | 因果中介分析 |
| interactions | 交互效应可视化 |
| emmeans | 边际均值估计 |

### 数据处理

| 包名 | 用途 |
|------|------|
| tidyverse | 数据科学工具集 |
| haven | SPSS/Stata/SAS 读取 |
| readxl | Excel 读取 |
| writexl | Excel 写入 |
| mice | 缺失数据填补 |

---

## 使用方法

### 方法 1: 便捷脚本 (推荐)

```bash
# 构建镜像
./r-stat.sh build

# 运行脚本
./r-stat.sh run my_analysis.R

# 交互式会话
./r-stat.sh shell
```

### 方法 2: 直接 Docker 命令

```bash
# 构建
docker build -t r-statistical:1.0 .

# 运行脚本
docker run --rm -v "$(pwd)":/workspace -w /workspace r-statistical:1.0 Rscript analysis.R

# 交互式 R
docker run --rm -it -v "$(pwd)":/workspace -w /workspace r-statistical:1.0 R

# 执行代码
docker run --rm r-statistical:1.0 Rscript -e 'print(1+1)'
```

### 方法 3: Docker Compose

创建 `docker-compose.yml`:

```yaml
version: '3.8'
services:
  r-stat:
    build: .
    volumes:
      - ./data:/workspace/data
      - ./scripts:/workspace/scripts
      - ./output:/workspace/output
    working_dir: /workspace
    command: Rscript scripts/analysis.R
```

运行:
```bash
docker-compose up
```

---

## 目录挂载

工作目录 `/workspace` 会挂载到当前目录，文件结构建议：

```
project/
├── data/           # 数据文件
│   ├── raw.xlsx
│   └── cleaned.csv
├── scripts/        # R 脚本
│   ├── 01_cleaning.R
│   ├── 02_descriptive.R
│   └── 03_analysis.R
├── output/         # 输出结果
│   ├── tables/
│   └── figures/
└── docker-compose.yml
```

---

## 中文支持

镜像已预装中文字体:
- Noto CJK (思源黑体)
- WenQuanYi Micro Hei (文泉驿微米黑)

在 R 脚本中设置中文字体:

```r
library(ggplot2)
library(showtext)

# 启用 showtext
showtext_auto()

# 使用 Noto Sans CJK
font_add("noto", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

# 绑定字体
theme_set(theme_bw(base_family = "noto"))

# 示例图表
ggplot(data, aes(x, y)) +
  geom_point() +
  labs(title = "中文标题", x = "横轴", y = "纵轴")
```

---

## 常见问题

### Q: 镜像构建失败?

**检查 Docker 状态**:
```bash
docker info
```

**清理缓存重建**:
```bash
docker build --no-cache -t r-statistical:1.0 .
```

### Q: 包安装失败?

**进入容器手动安装**:
```bash
./r-stat.sh bash
# 在容器中
R -e 'install.packages("packagename")'
```

### Q: 中文乱码?

**检查字体**:
```r
library(systemfonts)
system_fonts() |> filter(grepl("CJK|Noto|WenQuanYi", family))
```

**强制 UTF-8**:
```r
Sys.setlocale("LC_ALL", "C.UTF-8")
```

### Q: 如何持久化安装额外的包?

**方法 1**: 修改 Dockerfile 并重新构建

**方法 2**: 创建扩展 Dockerfile
```dockerfile
FROM r-statistical:1.0
RUN install2.r --error mypackage
```

---

## 资源占用

| 资源 | 大小 |
|------|------|
| 镜像大小 | ~3.5GB |
| 构建时间 | ~15-20分钟 |
| 内存需求 | 建议 4GB+ |

---

## 更新日志

### v1.0 (2025-02)
- 初始版本
- 基于 rocker/tidyverse:4.3
- 包含 40+ 统计分析包
- 支持 SEM/HLM/IRT/元分析
- 中文字体支持

---

## 许可证

MIT License
