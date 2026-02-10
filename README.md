# Claude Statistical Analysis Skill

A Claude Code skill that transforms Claude into a professional statistical consultant. Instead of blindly running whatever method the user requests, it **diagnoses first, then analyzes** — checking data quality, validating assumptions, and automatically selecting the appropriate statistical method.

### What It Does

When you upload a dataset and describe your analysis needs, Claude will:

1. **Data Profile** (automatic) — Sample size, variable types, missing patterns, distributions, outliers
2. **Assumption Checking** — Normality, homogeneity of variance, multicollinearity; auto-switches methods when assumptions fail
3. **Execute Analysis** — Runs the appropriate statistical method with proper controls
4. **Output Triple** — Every analysis produces three deliverables:
   - APA 7th edition table (Excel + Markdown)
   - Publication-quality figure (300dpi PNG)
   - Results paragraph ready for manuscript (English/Chinese)

### Supported Methods

| Complexity | Methods | Workflow |
|-----------|---------|----------|
| Simple | Descriptive stats, t-test, chi-square, correlation, reliability | Fast path (0 confirmations) |
| Medium | Regression, ANOVA, moderation, mediation, ROC/AUC, survival | Light path (1 confirmation) |
| Complex | SEM/CFA, HLM, IRT, meta-analysis, RI-CLPM | Full path (3-4 confirmations) |
| Planning | Power analysis / sample size calculation | Dedicated path (no data needed) |

### Installation

#### As a Claude Code Skill

```bash
# Copy to your Claude skills directory
cp -r . ~/.claude/skills/statistical-analysis/
```

Claude Code will automatically detect and activate this skill when you:
- Upload a data file (.xlsx, .csv, .sav)
- Say "help me analyze" / "run statistical analysis"
- Mention a specific method (t-test, regression, SEM, etc.)

#### R Docker Environment (Optional)

For advanced methods that require R (SEM path diagrams, HLM, IRT):

```bash
cd docker/
chmod +x r-stat.sh
./r-stat.sh build    # Build the Docker image (~2GB)
./r-stat.sh test     # Verify installation
```

Pre-installed R packages: lavaan, lme4, metafor, mirt, psych, tidyverse, semPlot, effectsize, and 30+ more.

### Project Structure

```
.
├── SKILL.md                    # Core skill definition (Claude reads this)
├── references/
│   ├── methods-index.md        # Method selection decision trees
│   ├── code-patterns.md        # Python/R code templates
│   ├── table-formats.md        # APA table format specifications
│   └── full-workflow.md        # Full path detailed workflow
├── docker/
│   ├── Dockerfile              # R environment (rocker/tidyverse + 40 packages)
│   ├── r-stat.sh               # Convenience script for Docker operations
│   ├── README.md               # Docker setup guide
│   └── examples/               # SEM, HLM, meta-analysis R examples
└── assets/
    └── report-template.md      # Analysis report template
```

### Key Design Decisions

#### "Diagnose Before Analyze"

The v3 approach was "user says do X, we do X." The v4 approach is "check the data and assumptions first, then decide what to do." This prevents common mistakes like:

- Running parametric tests on non-normal data
- Missing multicollinearity in regression
- Using wrong effect size measures
- Ignoring missing data patterns

#### Automatic Method Switching

When assumptions fail, the skill automatically switches to the appropriate alternative and informs the user:

```
> Note: Variable X failed the normality test (Shapiro-Wilk p = .003),
> automatically switched to Mann-Whitney U test (non-parametric alternative).
```

#### Output Triple Enforcement

Every analysis **must** produce all three outputs (table + figure + paragraph). A built-in `check_output_triplet()` mechanism ensures nothing is skipped.

### Examples

**User**: "I have survey data, please analyze the relationship between gaming addiction and mental health, controlling for demographics."

**Claude will**:
1. Run data profile (N, distributions, missing patterns)
2. Check: normality, VIF, outliers
3. Recommend: hierarchical regression (Step 1: demographics, Step 2: gaming addiction)
4. Execute with HC3 robust standard errors (if heteroscedasticity detected)
5. Output: regression table + coefficient forest plot + APA results paragraph

### Requirements

- **Claude Code** with skills support
- **Python**: pandas, numpy, scipy, statsmodels, matplotlib, seaborn (installed automatically)
- **R Docker** (optional): Docker Desktop for advanced methods

---

## License

MIT License - see [LICENSE](LICENSE)

## Changelog

- **v4.1** (2025-02-09): Mandatory output triple — figure generation embedded in all workflow paths
- **v4.0** (2025-02-09): "Diagnose before analyze" — data profiling, assumption checking, APA paragraph generation, medical research methods
- **v3.0** (2025-02-02): Three-tier path system, R Docker environment, APA table specifications
