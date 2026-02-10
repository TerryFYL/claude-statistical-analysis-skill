# ============================================================
# 多层线性模型 (HLM) 示例
#
# 运行: ./r-stat.sh run examples/hlm_example.R
# 经典图表: 个体轨迹图、随机效应图 (自动输出)
# ============================================================

library(lme4)
library(lmerTest)
library(performance)
library(tidyverse)

cat("=== 多层线性模型分析 ===\n\n")

# 使用内置数据集
data("sleepstudy", package = "lme4")
df <- sleepstudy

cat(sprintf("观测数: N = %d, 被试数: %d\n", nrow(df), length(unique(df$Subject))))

# ============================================================
# 1. 零模型 - ICC
# ============================================================

model_null <- lmer(Reaction ~ 1 + (1|Subject), data = df)
icc_val <- icc(model_null)

cat("\n【ICC】\n")
cat(sprintf("  ICC = %.3f (%.1f%% 方差来自个体间差异)\n",
            icc_val$ICC_adjusted, icc_val$ICC_adjusted * 100))

# ============================================================
# 2. 随机斜率模型
# ============================================================

model <- lmer(Reaction ~ Days + (1 + Days|Subject), data = df)

cat("\n【固定效应】\n")
fe <- fixef(model)
se <- sqrt(diag(vcov(model)))
summ <- summary(model)$coefficients
cat(sprintf("  截距: B = %.2f, SE = %.2f, t = %.2f, p < .001\n",
            fe[1], se[1], summ[1, "t value"]))
cat(sprintf("  Days: B = %.2f, SE = %.2f, t = %.2f, p < .001\n",
            fe[2], se[2], summ[2, "t value"]))

cat("\n【随机效应】\n")
vc <- VarCorr(model)$Subject
cat(sprintf("  截距方差 τ₀₀ = %.2f\n", vc[1,1]))
cat(sprintf("  斜率方差 τ₁₁ = %.2f\n", vc[2,2]))
cat(sprintf("  相关 r = %.2f\n", attr(vc, "correlation")[1,2]))
cat(sprintf("  残差方差 σ² = %.2f\n", sigma(model)^2))

# ============================================================
# 3. 经典图表输出
# ============================================================

cat("\n【图表输出】\n")

# 图1: 个体轨迹图
pdf("individual_trajectories.pdf", width = 10, height = 8)
df$predicted <- predict(model)
ggplot(df, aes(x = Days, y = Reaction, group = Subject)) +
  geom_line(aes(y = predicted), color = "steelblue", alpha = 0.6) +
  geom_point(alpha = 0.4, size = 1) +
  geom_smooth(aes(group = NULL), method = "lm", color = "red", linewidth = 1.5, se = FALSE) +
  labs(x = "Days", y = "Reaction Time (ms)",
       title = "Individual Trajectories with Fixed Effect") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()
cat("  ✓ individual_trajectories.pdf\n")

# 图2: 随机效应图
pdf("random_effects.pdf", width = 10, height = 8)
re <- ranef(model)$Subject
re$Subject <- rownames(re)
re_long <- re %>%
  pivot_longer(cols = c("(Intercept)", "Days"),
               names_to = "Effect", values_to = "Value")
ggplot(re_long, aes(x = reorder(Subject, Value), y = Value)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(color = "steelblue", size = 2) +
  facet_wrap(~Effect, scales = "free_y") +
  coord_flip() +
  labs(x = "Subject", y = "Random Effect",
       title = "Random Effects by Subject") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()
cat("  ✓ random_effects.pdf\n")

cat("\n=== 完成 ===\n")
