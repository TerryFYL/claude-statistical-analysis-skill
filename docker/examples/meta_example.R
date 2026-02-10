# ============================================================
# 元分析示例
#
# 运行: ./r-stat.sh run examples/meta_example.R
# 经典图表: 森林图、漏斗图 (自动输出)
# ============================================================

library(metafor)
library(tidyverse)

cat("=== 元分析 ===\n\n")

# 使用内置数据集
data("dat.bcg", package = "metafor")
df <- dat.bcg

cat(sprintf("研究数量: k = %d\n", nrow(df)))

# ============================================================
# 1. 效应量计算
# ============================================================

df <- escalc(measure = "RR",
             ai = tpos, bi = tneg,
             ci = cpos, di = cneg,
             data = df)

# ============================================================
# 2. 随机效应模型
# ============================================================

res <- rma(yi, vi, data = df, method = "REML")

cat("\n【整体效应】\n")
cat(sprintf("  RR = %.3f [95%% CI: %.3f, %.3f]\n",
            exp(res$beta), exp(res$ci.lb), exp(res$ci.ub)))
cat(sprintf("  z = %.2f, p = %.4f\n", res$zval, res$pval))

cat("\n【异质性】\n")
cat(sprintf("  Q(%d) = %.2f, p = %.4f\n", res$k - 1, res$QE, res$QEp))
cat(sprintf("  I² = %.1f%%\n", res$I2))
cat(sprintf("  τ² = %.4f\n", res$tau2))

# ============================================================
# 3. 发表偏倚
# ============================================================

cat("\n【发表偏倚检验】\n")
egger <- regtest(res)
cat(sprintf("  Egger's test: z = %.2f, p = %.3f\n", egger$zval, egger$pval))

taf <- trimfill(res)
if (taf$k0 > 0) {
  cat(sprintf("  Trim-and-Fill: 估计缺失 %d 篇\n", taf$k0))
  cat(sprintf("  校正后 RR = %.3f [%.3f, %.3f]\n",
              exp(taf$beta), exp(taf$ci.lb), exp(taf$ci.ub)))
} else {
  cat("  Trim-and-Fill: 未检测到缺失研究\n")
}

# ============================================================
# 4. 经典图表输出
# ============================================================

cat("\n【图表输出】\n")

# 森林图
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

# 漏斗图
pdf("funnel_plot.pdf", width = 8, height = 6)
funnel(res, main = "Funnel Plot")
dev.off()
cat("  ✓ funnel_plot.pdf\n")

cat("\n=== 完成 ===\n")
