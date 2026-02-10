#!/bin/bash
# ============================================================
# r-stat.sh - Statistical Analysis R Environment Convenience Script
#
# Usage:
#   ./r-stat.sh build          # Build image
#   ./r-stat.sh run script.R   # Run R script
#   ./r-stat.sh shell          # Enter interactive R
#   ./r-stat.sh bash           # Enter container bash
#   ./r-stat.sh test           # Run environment tests
# ============================================================

set -e

IMAGE_NAME="r-statistical"
IMAGE_TAG="1.0"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
}

# Check if the image exists
image_exists() {
    docker image inspect "$FULL_IMAGE" &> /dev/null
}

# Build image
cmd_build() {
    log_info "Building image: $FULL_IMAGE"
    docker build -t "$FULL_IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
    log_info "Build complete"
}

# Run R script
cmd_run() {
    if [ -z "$1" ]; then
        log_error "Please specify an R script path"
        echo "Usage: $0 run <script.R> [args...]"
        exit 1
    fi

    local script="$1"
    shift

    if [ ! -f "$script" ]; then
        log_error "File does not exist: $script"
        exit 1
    fi

    # Get the absolute path and directory of the script
    local abs_script=$(cd "$(dirname "$script")" && pwd)/$(basename "$script")
    local script_dir=$(dirname "$abs_script")
    local script_name=$(basename "$abs_script")

    log_info "Running script: $script_name"

    docker run --rm \
        -v "$script_dir":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        Rscript "$script_name" "$@"
}

# Execute R code string
cmd_eval() {
    if [ -z "$1" ]; then
        log_error "Please specify R code"
        echo "Usage: $0 eval '<R code>'"
        exit 1
    fi

    log_info "Executing R code"
    docker run --rm \
        -v "$(pwd)":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        Rscript -e "$1"
}

# Interactive R shell
cmd_shell() {
    log_info "Starting interactive R (exit: q())"
    docker run --rm -it \
        -v "$(pwd)":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        R --quiet
}

# Enter container bash
cmd_bash() {
    log_info "Entering container bash (exit: exit)"
    docker run --rm -it \
        -v "$(pwd)":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        bash
}

# Run environment tests
cmd_test() {
    log_info "Running environment tests..."

    docker run --rm "$FULL_IMAGE" Rscript -e '
cat("============================================================\n")
cat("Statistical Analysis R Environment Test\n")
cat("============================================================\n\n")

# Test core packages
test_packages <- list(
    "Structural Equation Modeling" = c("lavaan", "semPlot", "semTools"),
    "Hierarchical Linear Modeling" = c("lme4", "lmerTest"),
    "Meta-Analysis" = c("metafor", "meta"),
    "Item Response Theory" = c("mirt", "ltm"),
    "Psychometrics" = c("psych", "GPArotation"),
    "Data Processing" = c("tidyverse", "haven", "readxl"),
    "Effect Size Reporting" = c("effectsize", "parameters", "performance"),
    "Visualization" = c("ggplot2", "ggpubr", "corrplot")
)

total <- 0
passed <- 0

for (category in names(test_packages)) {
    cat(sprintf("\n[%s]\n", category))
    for (pkg in test_packages[[category]]) {
        total <- total + 1
        if (require(pkg, character.only = TRUE, quietly = TRUE)) {
            cat(sprintf("  ✓ %s (%s)\n", pkg, packageVersion(pkg)))
            passed <- passed + 1
        } else {
            cat(sprintf("  ✗ %s (not installed)\n", pkg))
        }
    }
}

cat("\n============================================================\n")
cat(sprintf("Test results: %d/%d packages available\n", passed, total))
cat("============================================================\n")

# Simple functional tests
cat("\n[Functional Tests]\n")

# lavaan test
tryCatch({
    suppressWarnings({
        model <- "y ~ x"
        fit <- lavaan::sem(model, data = data.frame(x = rnorm(50), y = rnorm(50)))
    })
    cat("  ✓ lavaan SEM available\n")
}, error = function(e) cat("  ✗ lavaan test failed\n"))

# lme4 test
tryCatch({
    suppressWarnings({
        data <- data.frame(y = rnorm(100), x = rnorm(100), g = rep(1:10, each = 10))
        fit <- lme4::lmer(y ~ x + (1|g), data = data)
    })
    cat("  ✓ lme4 HLM available\n")
}, error = function(e) cat("  ✗ lme4 test failed\n"))

# metafor test
tryCatch({
    dat <- metafor::escalc(measure = "RR", ai = 10, bi = 90, ci = 20, di = 80, n1i = 100, n2i = 100)
    cat("  ✓ metafor meta-analysis available\n")
}, error = function(e) cat("  ✗ metafor test failed\n"))

cat("\nTests complete!\n")
'
}

# Display help
cmd_help() {
    cat << EOF
Statistical Analysis R Environment Convenience Script

Usage: $0 <command> [options]

Commands:
  build         Build Docker image
  run <file>    Run R script
  eval '<code>' Execute R code string
  shell         Start interactive R
  bash          Enter container bash
  test          Run environment tests
  help          Show this help

Examples:
  $0 build                    # Build image before first use
  $0 run analysis.R           # Run analysis script
  $0 eval 'print(1+1)'        # Execute simple code
  $0 shell                    # Interactive R session

EOF
}

# Main entry point
main() {
    check_docker

    local cmd="${1:-help}"
    shift || true

    # Auto-build image if it doesn't exist
    if [ "$cmd" != "build" ] && [ "$cmd" != "help" ]; then
        if ! image_exists; then
            log_warn "Image not found, starting build..."
            cmd_build
        fi
    fi

    case "$cmd" in
        build)  cmd_build ;;
        run)    cmd_run "$@" ;;
        eval)   cmd_eval "$@" ;;
        shell)  cmd_shell ;;
        bash)   cmd_bash ;;
        test)   cmd_test ;;
        help)   cmd_help ;;
        *)
            log_error "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
