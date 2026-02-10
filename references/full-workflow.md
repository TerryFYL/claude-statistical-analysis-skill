# Full Path Workflow

**Applicable to**: SEM/CFA, HLM, IRT, Meta-analysis, RI-CLPM, and other complex analyses

---

## Workflow Overview

```
Phase 1: Data Cleaning Plan -> Pause for Confirmation
         |
Phase 2: Execute Cleaning -> Pause for Confirmation
         |
Phase 3: Analysis Plan -> Pause for Confirmation
         |
Phase 4: Execute Analysis -> Output Results
```

---

## Phase 1: Data Cleaning Plan

### Output Template

```markdown
## Data Overview
- Sample size: N = 3248
- Number of variables: 15

## Cleaning Plan

### Missing Value Treatment
| Variable | Missing Count | Missing % | Treatment |
|----------|---------------|-----------|-----------|
| Variable A | 50 | 5% | Deletion |
| Variable B | 300 | 30% | Mean imputation |

### Outlier Treatment
| Variable | Outlier Count | Treatment |
|----------|---------------|-----------|
| Age | 5 | Retain |

### Expected Results
- Expected final sample size: N = 3005
- Retention rate: 92.5%

---
Pause: Please confirm the cleaning plan
```

---

## Phase 2: Execute Cleaning

### Output Template

```markdown
## Cleaning Results
- Original: 3248 -> Final: 3005
- Retention rate: 92.5%

## Processing Details
1. Missing value deletion: 243 cases
2. Outlier treatment: 0 cases deleted

## Output Files
- data_cleaned.xlsx

---
Pause: Please confirm the cleaning results
```

---

## Phase 3: Analysis Plan

### Output Template

```markdown
## Variable Roles
| Role | Variable | Description |
|------|----------|-------------|
| Latent Variable 1 | visual | x1, x2, x3 |
| Latent Variable 2 | textual | x4, x5, x6 |
| Structural Path | visual -> textual | Hypothesized relationship |

## Analysis Plan
| No. | Content | Method | Output |
|-----|---------|--------|--------|
| 1 | CFA | lavaan | Path diagram, fit indices |
| 2 | SEM | lavaan | Path coefficients, model comparison |

## Output Specifications
- Chart language: Chinese/English?
- Table format: APA 7

---
Pause: Please confirm the analysis plan
```

---

## Phase 4: Execute Analysis

### R Code Execution

```bash
# Write script
cat > analysis.R << 'EOF'
library(lavaan)
library(semPlot)

data <- read.csv("data_clean.csv")

model <- '
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  textual ~ visual
'

fit <- sem(model, data = data)

# Output fit indices
cat("\n[Model Fit]\n")
fitmeasures(fit, c("cfi", "tli", "rmsea", "srmr"))

# Output path coefficients
cat("\n[Path Coefficients]\n")
parameterEstimates(fit, standardized = TRUE)

# Save path diagram
pdf("path_diagram.pdf", width = 10, height = 8)
semPaths(fit, what = "std", layout = "tree2")
dev.off()
EOF

# Execute
docker run --rm -v "$(pwd)":/workspace -w /workspace r-statistical:1.0 Rscript analysis.R
```

### Output Checklist

The following must be produced after analysis is complete:
- [ ] Results table (.xlsx)
- [ ] Charts (.pdf)
- [ ] Brief interpretation

---

## Handling Changes

If the user requests changes in a later phase:

```markdown
## Warning: Requirement Change

| Item | Original | New |
|------|----------|-----|
| [Item] | [Original] | [New] |

Proceed with the new requirements?
```
