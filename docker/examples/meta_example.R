# ============================================================
# Meta-Analysis Example
#
# Run: ./r-stat.sh run examples/meta_example.R
# Classic charts: Forest plot, funnel plot (auto output)
# ============================================================

library(metafor)
library(tidyverse)

cat("=== Meta-Analysis ===\n\n")

# Use built-in dataset
data("dat.bcg", package = "metafor")
df <- dat.bcg

cat(sprintf("Number of studies: k = %d\n", nrow(df)))

# ============================================================
# 1. Effect Size Calculation
# ============================================================

df <- escalc(measure = "RR",
             ai = tpos, bi = tneg,
             ci = cpos, di = cneg,
             data = df)

# ============================================================
# 2. Random-Effects Model
# ============================================================

res <- rma(yi, vi, data = df, method = "REML")

cat("\n[Overall Effect]\n")
cat(sprintf("  RR = %.3f [95%% CI: %.3f, %.3f]\n",
            exp(res$beta), exp(res$ci.lb), exp(res$ci.ub)))
cat(sprintf("  z = %.2f, p = %.4f\n", res$zval, res$pval))

cat("\n[Heterogeneity]\n")
cat(sprintf("  Q(%d) = %.2f, p = %.4f\n", res$k - 1, res$QE, res$QEp))
cat(sprintf("  I^2 = %.1f%%\n", res$I2))
cat(sprintf("  tau^2 = %.4f\n", res$tau2))

# ============================================================
# 3. Publication Bias
# ============================================================

cat("\n[Publication Bias Tests]\n")
egger <- regtest(res)
cat(sprintf("  Egger's test: z = %.2f, p = %.3f\n", egger$zval, egger$pval))

taf <- trimfill(res)
if (taf$k0 > 0) {
  cat(sprintf("  Trim-and-Fill: estimated %d missing studies\n", taf$k0))
  cat(sprintf("  Adjusted RR = %.3f [%.3f, %.3f]\n",
              exp(taf$beta), exp(taf$ci.lb), exp(taf$ci.ub)))
} else {
  cat("  Trim-and-Fill: no missing studies detected\n")
}

# ============================================================
# 4. Classic Chart Output
# ============================================================

cat("\n[Chart Output]\n")

# Forest plot
pdf("forest_plot.pdf", width = 12, height = 8)
forest(res,
       slab = paste0(df$author, " (", df$year, ")"),
       atransf = exp,
       at = log(c(0.05, 0.1, 0.25, 0.5, 1, 2)),
       xlim = c(-4, 3),
       xlab = "Risk Ratio",
       header = c("Study", "RR [95% CI]"))
dev.off()
cat("  ✓ forest_plot.pdf\n")

# Funnel plot
pdf("funnel_plot.pdf", width = 8, height = 6)
funnel(res, main = "Funnel Plot")
dev.off()
cat("  ✓ funnel_plot.pdf\n")

cat("\n=== Complete ===\n")
