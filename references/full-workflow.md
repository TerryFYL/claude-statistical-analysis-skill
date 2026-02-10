# 完整路径工作流程

**适用**: SEM/CFA、HLM、IRT、元分析、RI-CLPM 等复杂分析

---

## 流程概览

```
阶段1: 数据清洗方案 → ⏸️确认
         ↓
阶段2: 执行清洗 → ⏸️确认
         ↓
阶段3: 分析方案 → ⏸️确认
         ↓
阶段4: 执行分析 → 输出结果
```

---

## 阶段1: 数据清洗方案

### 输出模板

```markdown
## 数据概况
- 样本量: N = 3248
- 变量数: 15

## 清洗方案

### 缺失值处理
| 变量 | 缺失数 | 缺失% | 处理方式 |
|------|--------|-------|----------|
| 变量A | 50 | 5% | 删除 |
| 变量B | 300 | 30% | 均值填补 |

### 异常值处理
| 变量 | 异常数 | 处理方式 |
|------|--------|----------|
| 年龄 | 5 | 保留 |

### 预计结果
- 预计最终样本量: N = 3005
- 保留率: 92.5%

---
⏸️ 请确认清洗方案
```

---

## 阶段2: 执行清洗

### 输出模板

```markdown
## 清洗结果
- 原始: 3248 → 最终: 3005
- 保留率: 92.5%

## 处理详情
1. 缺失值删除: 243例
2. 异常值处理: 0例删除

## 输出文件
- 数据_清洗后.xlsx

---
⏸️ 请确认清洗结果
```

---

## 阶段3: 分析方案

### 输出模板

```markdown
## 变量角色
| 角色 | 变量 | 说明 |
|------|------|------|
| 潜变量1 | visual | x1, x2, x3 |
| 潜变量2 | textual | x4, x5, x6 |
| 结构路径 | visual → textual | 假设关系 |

## 分析计划
| 序号 | 内容 | 方法 | 输出 |
|------|------|------|------|
| 1 | CFA | lavaan | 路径图、拟合指标 |
| 2 | SEM | lavaan | 路径系数、模型比较 |

## 输出规范
- 图表语言: 中文/英文？
- 表格格式: APA 7

---
⏸️ 请确认分析方案
```

---

## 阶段4: 执行分析

### R 代码执行

```bash
# 写入脚本
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

# 输出拟合指标
cat("\n【模型拟合】\n")
fitmeasures(fit, c("cfi", "tli", "rmsea", "srmr"))

# 输出路径系数
cat("\n【路径系数】\n")
parameterEstimates(fit, standardized = TRUE)

# 保存路径图
pdf("path_diagram.pdf", width = 10, height = 8)
semPaths(fit, what = "std", layout = "tree2")
dev.off()
EOF

# 执行
docker run --rm -v "$(pwd)":/workspace -w /workspace r-statistical:1.0 Rscript analysis.R
```

### 输出清单

分析完成后必须输出：
- [ ] 结果表格 (.xlsx)
- [ ] 图表 (.pdf)
- [ ] 简要解读

---

## 变更处理

如果用户在后续阶段提出变更：

```markdown
## ⚠️ 需求变更

| 项目 | 原内容 | 新内容 |
|------|--------|--------|
| [项目] | [原] | [新] |

是否按新需求执行？
```
