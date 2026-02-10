# ============================================================
# Statistical Analysis R Environment
#
# Base: rocker/tidyverse:4.3
# Purpose: 社会科学/心理学统计分析专用环境
# Includes: lavaan, lme4, metafor, mirt, psych, and more
# ============================================================

FROM rocker/tidyverse:latest

LABEL maintainer="Statistical Analysis Skill"
LABEL description="R environment for advanced statistical analysis (SEM, HLM, IRT, Meta-analysis)"
LABEL version="1.0"

# ============================================================
# 1. 系统依赖
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 中文字体支持
    fonts-noto-cjk \
    fonts-wqy-microhei \
    # LaTeX 支持 (用于高质量图表导出)
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    # 图形库依赖
    libglpk-dev \
    # 系统工具
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 刷新字体缓存
RUN fc-cache -fv

# ============================================================
# 2. R 包安装 - 核心统计分析
# ============================================================
RUN install2.r --error --skipinstalled --ncpus -1 \
    # === 结构方程模型 (SEM) ===
    lavaan \
    semPlot \
    semTools \
    # === 多层线性模型 (HLM) ===
    lme4 \
    lmerTest \
    nlme \
    # === 元分析 ===
    metafor \
    meta \
    # === 项目反应理论 (IRT) ===
    mirt \
    ltm \
    # === 心理测量学 ===
    psych \
    GPArotation \
    nFactors \
    # === 效应量与报告 ===
    effectsize \
    parameters \
    performance \
    see \
    report \
    # === 中介与调节分析 ===
    mediation \
    interactions \
    # === 方差分析与事后检验 ===
    emmeans \
    afex \
    # === 缺失数据处理 ===
    mice \
    naniar \
    # === 可视化增强 ===
    ggpubr \
    corrplot \
    ggcorrplot \
    sjPlot \
    && rm -rf /tmp/downloaded_packages/

# ============================================================
# 3. R 包安装 - 数据处理与工具
# ============================================================
RUN install2.r --error --skipinstalled --ncpus -1 \
    # === 数据导入 ===
    haven \
    readxl \
    writexl \
    openxlsx \
    foreign \
    # === 数据处理 ===
    data.table \
    janitor \
    # === 表格输出 ===
    gt \
    flextable \
    kableExtra \
    stargazer \
    # === 工具函数 ===
    broom \
    broom.mixed \
    insight \
    && rm -rf /tmp/downloaded_packages/

# ============================================================
# 4. 创建工作目录
# ============================================================
WORKDIR /workspace

# ============================================================
# 5. R 环境配置
# ============================================================
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >> /usr/local/lib/R/etc/Rprofile.site \
    && echo 'options(warn = 1)' >> /usr/local/lib/R/etc/Rprofile.site \
    && echo 'Sys.setlocale("LC_ALL", "C.UTF-8")' >> /usr/local/lib/R/etc/Rprofile.site

# ============================================================
# 6. 验证安装
# ============================================================
RUN Rscript -e 'cat("=== Installed Packages ===\n"); \
    core_pkgs <- c("lavaan", "lme4", "metafor", "mirt", "psych", "tidyverse"); \
    for(pkg in core_pkgs) { \
        if(require(pkg, character.only=TRUE, quietly=TRUE)) { \
            cat(sprintf("✓ %s: %s\n", pkg, packageVersion(pkg))); \
        } else { \
            cat(sprintf("✗ %s: NOT INSTALLED\n", pkg)); \
        } \
    }'

# 默认命令
CMD ["R"]
