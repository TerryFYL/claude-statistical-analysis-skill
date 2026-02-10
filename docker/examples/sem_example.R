# ============================================================
# 结构方程模型 (SEM) 示例
#
# 运行: ./r-stat.sh run examples/sem_example.R
# 经典图表: 路径图 (自动输出)
# ============================================================

library(lavaan)
library(semPlot)
library(tidyverse)

cat("=== 结构方程模型分析 ===\n\n")

# 使用内置数据集
data("HolzingerSwineford1939")
df <- HolzingerSwineford1939

cat(sprintf("样本量: N = %d\n", nrow(df)))

# ============================================================
# 1. 验证性因子分析 (CFA)
# ============================================================

cfa_model <- '
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
'

fit_cfa <- cfa(cfa_model, data = df)

# 核心拟合指标
cat("\n【模型拟合】\n")
fit_idx <- fitMeasures(fit_cfa, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
cat(sprintf("  χ²(%d) = %.2f, p = %.3f\n", fit_idx["df"], fit_idx["chisq"], fit_idx["pvalue"]))
cat(sprintf("  CFI = %.3f, TLI = %.3f\n", fit_idx["cfi"], fit_idx["tli"]))
cat(sprintf("  RMSEA = %.3f, SRMR = %.3f\n", fit_idx["rmsea"], fit_idx["srmr"]))

# 因子载荷
cat("\n【标准化因子载荷】\n")
params <- parameterEstimates(fit_cfa, standardized = TRUE)
loadings <- params[params$op == "=~", c("lhs", "rhs", "std.all", "pvalue")]
for (i in 1:nrow(loadings)) {
  sig <- ifelse(loadings$pvalue[i] < 0.001, "***",
         ifelse(loadings$pvalue[i] < 0.01, "**",
         ifelse(loadings$pvalue[i] < 0.05, "*", "")))
  cat(sprintf("  %s → %s: λ = %.3f %s\n",
              loadings$lhs[i], loadings$rhs[i], loadings$std.all[i], sig))
}

# ============================================================
# 2. 经典图表输出 - 路径图
# ============================================================

cat("\n【图表输出】\n")

# CFA 路径图
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

cat("\n=== 完成 ===\n")
