#!/bin/bash
# ============================================================
# r-stat.sh - 统计分析 R 环境便捷脚本
#
# 用法:
#   ./r-stat.sh build          # 构建镜像
#   ./r-stat.sh run script.R   # 运行 R 脚本
#   ./r-stat.sh shell          # 进入交互式 R
#   ./r-stat.sh bash           # 进入容器 bash
#   ./r-stat.sh test           # 运行环境测试
# ============================================================

set -e

IMAGE_NAME="r-statistical"
IMAGE_TAG="1.0"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
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

# 检查 Docker 是否可用
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon 未运行"
        exit 1
    fi
}

# 检查镜像是否存在
image_exists() {
    docker image inspect "$FULL_IMAGE" &> /dev/null
}

# 构建镜像
cmd_build() {
    log_info "构建镜像: $FULL_IMAGE"
    docker build -t "$FULL_IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
    log_info "构建完成"
}

# 运行 R 脚本
cmd_run() {
    if [ -z "$1" ]; then
        log_error "请指定 R 脚本路径"
        echo "用法: $0 run <script.R> [args...]"
        exit 1
    fi

    local script="$1"
    shift

    if [ ! -f "$script" ]; then
        log_error "文件不存在: $script"
        exit 1
    fi

    # 获取脚本的绝对路径和目录
    local abs_script=$(cd "$(dirname "$script")" && pwd)/$(basename "$script")
    local script_dir=$(dirname "$abs_script")
    local script_name=$(basename "$abs_script")

    log_info "运行脚本: $script_name"

    docker run --rm \
        -v "$script_dir":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        Rscript "$script_name" "$@"
}

# 执行 R 代码字符串
cmd_eval() {
    if [ -z "$1" ]; then
        log_error "请指定 R 代码"
        echo "用法: $0 eval '<R code>'"
        exit 1
    fi

    log_info "执行 R 代码"
    docker run --rm \
        -v "$(pwd)":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        Rscript -e "$1"
}

# 交互式 R shell
cmd_shell() {
    log_info "启动交互式 R (退出: q())"
    docker run --rm -it \
        -v "$(pwd)":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        R --quiet
}

# 进入容器 bash
cmd_bash() {
    log_info "进入容器 bash (退出: exit)"
    docker run --rm -it \
        -v "$(pwd)":/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        bash
}

# 运行环境测试
cmd_test() {
    log_info "运行环境测试..."

    docker run --rm "$FULL_IMAGE" Rscript -e '
cat("============================================================\n")
cat("Statistical Analysis R Environment Test\n")
cat("============================================================\n\n")

# 测试核心包
test_packages <- list(
    "结构方程模型" = c("lavaan", "semPlot", "semTools"),
    "多层线性模型" = c("lme4", "lmerTest"),
    "元分析" = c("metafor", "meta"),
    "项目反应理论" = c("mirt", "ltm"),
    "心理测量" = c("psych", "GPArotation"),
    "数据处理" = c("tidyverse", "haven", "readxl"),
    "效应量报告" = c("effectsize", "parameters", "performance"),
    "可视化" = c("ggplot2", "ggpubr", "corrplot")
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
            cat(sprintf("  ✗ %s (未安装)\n", pkg))
        }
    }
}

cat("\n============================================================\n")
cat(sprintf("测试结果: %d/%d 包可用\n", passed, total))
cat("============================================================\n")

# 简单功能测试
cat("\n[功能测试]\n")

# lavaan 测试
tryCatch({
    suppressWarnings({
        model <- "y ~ x"
        fit <- lavaan::sem(model, data = data.frame(x = rnorm(50), y = rnorm(50)))
    })
    cat("  ✓ lavaan SEM 可用\n")
}, error = function(e) cat("  ✗ lavaan 测试失败\n"))

# lme4 测试
tryCatch({
    suppressWarnings({
        data <- data.frame(y = rnorm(100), x = rnorm(100), g = rep(1:10, each = 10))
        fit <- lme4::lmer(y ~ x + (1|g), data = data)
    })
    cat("  ✓ lme4 HLM 可用\n")
}, error = function(e) cat("  ✗ lme4 测试失败\n"))

# metafor 测试
tryCatch({
    dat <- metafor::escalc(measure = "RR", ai = 10, bi = 90, ci = 20, di = 80, n1i = 100, n2i = 100)
    cat("  ✓ metafor 元分析可用\n")
}, error = function(e) cat("  ✗ metafor 测试失败\n"))

cat("\n测试完成!\n")
'
}

# 显示帮助
cmd_help() {
    cat << EOF
统计分析 R 环境便捷脚本

用法: $0 <command> [options]

命令:
  build         构建 Docker 镜像
  run <file>    运行 R 脚本
  eval '<code>' 执行 R 代码字符串
  shell         启动交互式 R
  bash          进入容器 bash
  test          运行环境测试
  help          显示此帮助

示例:
  $0 build                    # 首次使用前构建镜像
  $0 run analysis.R           # 运行分析脚本
  $0 eval 'print(1+1)'        # 执行简单代码
  $0 shell                    # 交互式 R 会话

EOF
}

# 主入口
main() {
    check_docker

    local cmd="${1:-help}"
    shift || true

    # 自动构建镜像（如果不存在）
    if [ "$cmd" != "build" ] && [ "$cmd" != "help" ]; then
        if ! image_exists; then
            log_warn "镜像不存在，开始构建..."
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
            log_error "未知命令: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
