# ============================================================
# Hierarchical Linear Modeling (HLM) Example
#
# Run: ./r-stat.sh run examples/hlm_example.R
# Classic charts: Individual trajectory plot, random effects plot (auto output)
# ============================================================

library(lme4)
library(lmerTest)
library(performance)
library(tidyverse)

cat("=== Hierarchical Linear Modeling Analysis ===\n\n")

# Use built-in dataset
data("sleepstudy", package = "lme4")
df <- sleepstudy

cat(sprintf("Observations: N = %d, Subjects: %d\n", nrow(df), length(unique(df$Subject))))

# ============================================================
# 1. Null Model - ICC
# ============================================================

model_null <- lmer(Reaction ~ 1 + (1|Subject), data = df)
icc_val <- icc(model_null)

cat("\n[ICC]\n")
cat(sprintf("  ICC = %.3f (%.1f%% of variance is between-subject)\n",
            icc_val$ICC_adjusted, icc_val$ICC_adjusted * 100))

# ============================================================
# 2. Random Slope Model
# ============================================================

model <- lmer(Reaction ~ Days + (1 + Days|Subject), data = df)

cat("\n[Fixed Effects]\n")
fe <- fixef(model)
se <- sqrt(diag(vcov(model)))
summ <- summary(model)$coefficients
cat(sprintf("  Intercept: B = %.2f, SE = %.2f, t = %.2f, p < .001\n",
            fe[1], se[1], summ[1, "t value"]))
cat(sprintf("  Days: B = %.2f, SE = %.2f, t = %.2f, p < .001\n",
            fe[2], se[2], summ[2, "t value"]))

cat("\n[Random Effects]\n")
vc <- VarCorr(model)$Subject
cat(sprintf("  Intercept variance tau_00 = %.2f\n", vc[1,1]))
cat(sprintf("  Slope variance tau_11 = %.2f\n", vc[2,2]))
cat(sprintf("  Correlation r = %.2f\n", attr(vc, "correlation")[1,2]))
cat(sprintf("  Residual variance sigma^2 = %.2f\n", sigma(model)^2))

# ============================================================
# 3. Classic Chart Output
# ============================================================

cat("\n[Chart Output]\n")

# Figure 1: Individual trajectory plot
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

# Figure 2: Random effects plot
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

cat("\n=== Complete ===\n")
