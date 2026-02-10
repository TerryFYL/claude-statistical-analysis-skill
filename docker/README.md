# Statistical Analysis R Docker Environment

> A specialized R environment for social science/psychology statistical analysis, supporting advanced methods including SEM, HLM, meta-analysis, IRT, and more.

---

## Quick Start

### 1. Build Image

```bash
cd docker/
chmod +x r-stat.sh
./r-stat.sh build
```

### 2. Run Analysis

```bash
# Run an R script
./r-stat.sh run analysis.R

# Interactive R
./r-stat.sh shell

# Execute a code snippet
./r-stat.sh eval 'library(lavaan); packageVersion("lavaan")'
```

### 3. Verify Environment

```bash
./r-stat.sh test
```

---

## Included R Packages

### Structural Equation Modeling (SEM)

| Package | Version | Purpose |
|---------|---------|---------|
| lavaan | 0.6+ | Core SEM/CFA engine |
| semPlot | - | Path diagram visualization |
| semTools | - | SEM utility tools |

**Example**:
```r
library(lavaan)

model <- '
  # Measurement model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9

  # Structural model
  speed ~ visual + textual
'

fit <- sem(model, data = HolzingerSwineford1939)
summary(fit, fit.measures = TRUE, standardized = TRUE)
```

### Hierarchical Linear Modeling (HLM)

| Package | Purpose |
|---------|---------|
| lme4 | Core mixed-effects modeling |
| lmerTest | Provides p-values |
| nlme | Traditional mixed models |

**Example**:
```r
library(lme4)
library(lmerTest)

# Random intercept model
model <- lmer(score ~ treatment + (1|school), data = mydata)

# Random slope model
model <- lmer(score ~ treatment + (1 + treatment|school), data = mydata)

summary(model)
```

### Meta-Analysis

| Package | Purpose |
|---------|---------|
| metafor | Core meta-analysis |
| meta | Simplified interface |

**Example**:
```r
library(metafor)

# Effect size calculation
dat <- escalc(measure = "SMD",
              m1i = mean_exp, sd1i = sd_exp, n1i = n_exp,
              m2i = mean_ctrl, sd2i = sd_ctrl, n2i = n_ctrl,
              data = mydata)

# Random-effects model
res <- rma(yi, vi, data = dat, method = "REML")
summary(res)

# Forest plot
forest(res)
```

### Item Response Theory (IRT)

| Package | Purpose |
|---------|---------|
| mirt | Core IRT analysis |
| ltm | Traditional IRT models |

**Example**:
```r
library(mirt)

# 2PL model
fit <- mirt(response_matrix, 1, itemtype = "2PL")

# Item parameters
coef(fit, simplify = TRUE)

# Ability estimates
theta <- fscores(fit)

# Item characteristic curves
plot(fit, type = "trace")
```

### Psychometrics & Factor Analysis

| Package | Purpose |
|---------|---------|
| psych | Psychometrics toolkit |
| GPArotation | Factor rotation |
| nFactors | Determining number of factors |

### Effect Sizes & Reporting

| Package | Purpose |
|---------|---------|
| effectsize | Effect size calculation |
| parameters | Parameter extraction |
| performance | Model evaluation |
| report | Automated report generation |

### Mediation & Moderation

| Package | Purpose |
|---------|---------|
| mediation | Causal mediation analysis |
| interactions | Interaction effect visualization |
| emmeans | Estimated marginal means |

### Data Processing

| Package | Purpose |
|---------|---------|
| tidyverse | Data science toolkit |
| haven | Read SPSS/Stata/SAS files |
| readxl | Read Excel files |
| writexl | Write Excel files |
| mice | Multiple imputation for missing data |

---

## Usage

### Method 1: Convenience Script (Recommended)

```bash
# Build image
./r-stat.sh build

# Run script
./r-stat.sh run my_analysis.R

# Interactive session
./r-stat.sh shell
```

### Method 2: Direct Docker Commands

```bash
# Build
docker build -t r-statistical:1.0 .

# Run script
docker run --rm -v "$(pwd)":/workspace -w /workspace r-statistical:1.0 Rscript analysis.R

# Interactive R
docker run --rm -it -v "$(pwd)":/workspace -w /workspace r-statistical:1.0 R

# Execute code
docker run --rm r-statistical:1.0 Rscript -e 'print(1+1)'
```

### Method 3: Docker Compose

Create a `docker-compose.yml`:

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

Run:
```bash
docker-compose up
```

---

## Directory Mounting

The working directory `/workspace` is mounted to the current directory. Recommended file structure:

```
project/
├── data/           # Data files
│   ├── raw.xlsx
│   └── cleaned.csv
├── scripts/        # R scripts
│   ├── 01_cleaning.R
│   ├── 02_descriptive.R
│   └── 03_analysis.R
├── output/         # Output results
│   ├── tables/
│   └── figures/
└── docker-compose.yml
```

---

## Chinese Language Support

The image comes with pre-installed Chinese fonts:
- Noto CJK (Source Han Sans)
- WenQuanYi Micro Hei

To set Chinese fonts in R scripts:

```r
library(ggplot2)
library(showtext)

# Enable showtext
showtext_auto()

# Use Noto Sans CJK
font_add("noto", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

# Bind font
theme_set(theme_bw(base_family = "noto"))

# Example plot
ggplot(data, aes(x, y)) +
  geom_point() +
  labs(title = "Chinese Title", x = "X Axis", y = "Y Axis")
```

---

## FAQ

### Q: Image build failed?

**Check Docker status**:
```bash
docker info
```

**Clean cache and rebuild**:
```bash
docker build --no-cache -t r-statistical:1.0 .
```

### Q: Package installation failed?

**Enter the container and install manually**:
```bash
./r-stat.sh bash
# Inside the container
R -e 'install.packages("packagename")'
```

### Q: Chinese character encoding issues?

**Check fonts**:
```r
library(systemfonts)
system_fonts() |> filter(grepl("CJK|Noto|WenQuanYi", family))
```

**Force UTF-8**:
```r
Sys.setlocale("LC_ALL", "C.UTF-8")
```

### Q: How to persistently install additional packages?

**Method 1**: Modify the Dockerfile and rebuild

**Method 2**: Create an extended Dockerfile
```dockerfile
FROM r-statistical:1.0
RUN install2.r --error mypackage
```

---

## Resource Usage

| Resource | Size |
|----------|------|
| Image size | ~3.5GB |
| Build time | ~15-20 minutes |
| Memory requirement | Recommended 4GB+ |

---

## Changelog

### v1.0 (2025-02)
- Initial release
- Based on rocker/tidyverse:4.3
- Includes 40+ statistical analysis packages
- Supports SEM/HLM/IRT/Meta-analysis
- Chinese font support

---

## License

MIT License
