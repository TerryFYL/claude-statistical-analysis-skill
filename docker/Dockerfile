# ============================================================
# Statistical Analysis R Environment
#
# Base: rocker/tidyverse:4.3
# Purpose: Specialized environment for social science/psychology statistical analysis
# Includes: lavaan, lme4, metafor, mirt, psych, and more
# ============================================================

FROM rocker/tidyverse:latest

LABEL maintainer="Statistical Analysis Skill"
LABEL description="R environment for advanced statistical analysis (SEM, HLM, IRT, Meta-analysis)"
LABEL version="1.0"

# ============================================================
# 1. System Dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chinese font support
    fonts-noto-cjk \
    fonts-wqy-microhei \
    # LaTeX support (for high-quality chart export)
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    # Graphics library dependencies
    libglpk-dev \
    # System tools
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Refresh font cache
RUN fc-cache -fv

# ============================================================
# 2. R Package Installation - Core Statistical Analysis
# ============================================================
RUN install2.r --error --skipinstalled --ncpus -1 \
    # === Structural Equation Modeling (SEM) ===
    lavaan \
    semPlot \
    semTools \
    # === Hierarchical Linear Modeling (HLM) ===
    lme4 \
    lmerTest \
    nlme \
    # === Meta-Analysis ===
    metafor \
    meta \
    # === Item Response Theory (IRT) ===
    mirt \
    ltm \
    # === Psychometrics ===
    psych \
    GPArotation \
    nFactors \
    # === Effect Sizes & Reporting ===
    effectsize \
    parameters \
    performance \
    see \
    report \
    # === Mediation & Moderation Analysis ===
    mediation \
    interactions \
    # === ANOVA & Post-hoc Tests ===
    emmeans \
    afex \
    # === Missing Data Handling ===
    mice \
    naniar \
    # === Visualization Enhancement ===
    ggpubr \
    corrplot \
    ggcorrplot \
    sjPlot \
    && rm -rf /tmp/downloaded_packages/

# ============================================================
# 3. R Package Installation - Data Processing & Tools
# ============================================================
RUN install2.r --error --skipinstalled --ncpus -1 \
    # === Data Import ===
    haven \
    readxl \
    writexl \
    openxlsx \
    foreign \
    # === Data Processing ===
    data.table \
    janitor \
    # === Table Output ===
    gt \
    flextable \
    kableExtra \
    stargazer \
    # === Utility Functions ===
    broom \
    broom.mixed \
    insight \
    && rm -rf /tmp/downloaded_packages/

# ============================================================
# 4. Create Working Directory
# ============================================================
WORKDIR /workspace

# ============================================================
# 5. R Environment Configuration
# ============================================================
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >> /usr/local/lib/R/etc/Rprofile.site \
    && echo 'options(warn = 1)' >> /usr/local/lib/R/etc/Rprofile.site \
    && echo 'Sys.setlocale("LC_ALL", "C.UTF-8")' >> /usr/local/lib/R/etc/Rprofile.site

# ============================================================
# 6. Verify Installation
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

# Default command
CMD ["R"]
