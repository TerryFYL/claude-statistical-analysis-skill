# ============================================================
# Structural Equation Modeling (SEM) Example
#
# Run: ./r-stat.sh run examples/sem_example.R
# Classic chart: Path diagram (auto output)
# ============================================================

library(lavaan)
library(semPlot)
library(tidyverse)

cat("=== Structural Equation Modeling Analysis ===\n\n")

# Use built-in dataset
data("HolzingerSwineford1939")
df <- HolzingerSwineford1939

cat(sprintf("Sample size: N = %d\n", nrow(df)))

# ============================================================
# 1. Confirmatory Factor Analysis (CFA)
# ============================================================

cfa_model <- '
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
'

fit_cfa <- cfa(cfa_model, data = df)

# Core fit indices
cat("\n[Model Fit]\n")
fit_idx <- fitMeasures(fit_cfa, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
cat(sprintf("  chi2(%d) = %.2f, p = %.3f\n", fit_idx["df"], fit_idx["chisq"], fit_idx["pvalue"]))
cat(sprintf("  CFI = %.3f, TLI = %.3f\n", fit_idx["cfi"], fit_idx["tli"]))
cat(sprintf("  RMSEA = %.3f, SRMR = %.3f\n", fit_idx["rmsea"], fit_idx["srmr"]))

# Factor loadings
cat("\n[Standardized Factor Loadings]\n")
params <- parameterEstimates(fit_cfa, standardized = TRUE)
loadings <- params[params$op == "=~", c("lhs", "rhs", "std.all", "pvalue")]
for (i in 1:nrow(loadings)) {
  sig <- ifelse(loadings$pvalue[i] < 0.001, "***",
         ifelse(loadings$pvalue[i] < 0.01, "**",
         ifelse(loadings$pvalue[i] < 0.05, "*", "")))
  cat(sprintf("  %s -> %s: lambda = %.3f %s\n",
              loadings$lhs[i], loadings$rhs[i], loadings$std.all[i], sig))
}

# ============================================================
# 2. Classic Chart Output - Path Diagram
# ============================================================

cat("\n[Chart Output]\n")

# CFA path diagram
pdf("cfa_path_diagram.pdf", width = 10, height = 8)
semPaths(fit_cfa,
         what = "std",
         layout = "tree2",
         edge.label.cex = 1.2,
         sizeMan = 8,
         sizeLat = 10,
         style = "lisrel",
         nCharNodes = 7,
         edge.color = "black",
         color = list(lat = "lightblue", man = "lightyellow"),
         title = FALSE)
dev.off()
cat("  ✓ cfa_path_diagram.pdf\n")

cat("\n=== Complete ===\n")
