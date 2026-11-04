#!/usr/bin/env bash
# =============================================================================
# Titan-LM | Production Mega-Corpus Harvester
# Shallow clones massive Tier-1 C/C++/QML codebases into ~/titan_massive_dataset
# =============================================================================

set -euo pipefail

TARGET_DIR="${HOME}/titan_massive_dataset"

# ── ANSI Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log()   { echo -e "${CYAN}[$(date +'%H:%M:%S')] [INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[$(date +'%H:%M:%S')] [ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[$(date +'%H:%M:%S')] [WARN]${NC} $*"; }
err()   { echo -e "${RED}[$(date +'%H:%M:%S')] [FAIL]${NC} $*"; }

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║      Titan-LM  ·  Production Mega-Corpus Harvester           ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Initialize Dedicated Target Directory ────────────────────────────
log "Creating dedicated corpus directory: ${BOLD}${TARGET_DIR}${NC}"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"
ok "Target directory ready: $(pwd)"
echo ""

# ── Helper: Robust Shallow Clone ─────────────────────────────────────────────
clone_repo() {
    local name="$1"
    local url="$2"
    local desc="$3"
    local dest="${TARGET_DIR}/${name}"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}▶ Target:${NC} ${name}"
    echo -e "${BOLD}  Description:${NC} ${desc}"
    echo -e "${BOLD}  Source URL :${NC} ${url}"

    if [[ -d "${dest}/.git" || -d "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null)" ]]; then
        warn "Directory '${name}' already exists in target. Skipping clone to preserve bandwidth."
    else
        log "Cloning with --depth 1 (shallow, single-branch, no commit history)..."
        local t0
        t0=$(date +%s)

        if git clone --depth 1 --single-branch "${url}" "${dest}"; then
            local t1
            t1=$(date +%s)
            local elapsed=$((t1 - t0))
            local size
            size=$(du -sh "${dest}" 2>/dev/null | awk '{print $1}')
            ok "Successfully harvested ${BOLD}${name}${NC} (${size}) in ${elapsed}s"
        else
            err "Failed to clone ${name} from ${url}"
            return 1
        fi
    fi
    echo ""
}

# ── Step 2: Harvest Elite Tier-1 Codebases ───────────────────────────────────

# 1. LLVM Project: State of the art compiler infrastructure, libc++, AST, template metaprogramming
clone_repo \
    "llvm-project" \
    "https://github.com/llvm/llvm-project.git" \
    "LLVM, Clang, MLIR, libcxx - Elite template-heavy modern C++20 compiler engineering"

# 2. Linux Kernel: The standard for bare-metal systems, memory management, drivers, low-level concurrency
clone_repo \
    "linux" \
    "https://github.com/torvalds/linux.git" \
    "Torvalds Linux Kernel - Definitive standard for low-level systems C & hardware architecture"

# 3. TensorFlow: Deep learning computational graphs, XLA compiler, Eigen/SIMD kernels
clone_repo \
    "tensorflow" \
    "https://github.com/tensorflow/tensorflow.git" \
    "TensorFlow - Production C++ compute graphs, autograd operations, and tensor engines"

# 4. OpenCV: Matrix manipulation, computer vision, SIMD-accelerated linear algebra algorithms
clone_repo \
    "opencv" \
    "https://github.com/opencv/opencv.git" \
    "OpenCV - High-performance numerical linear algebra, image matrices, and SIMD kernels"

# 5. Qt5: Complete cross-platform desktop UI engine, declarative QML hierarchy, signal/slot architecture
clone_repo \
    "qt5" \
    "https://github.com/qt/qt5.git" \
    "Qt5 - Official framework engine for declarative QML components and modern C++ GUI systems"

# ── Step 3: Dataset Telemetry & File Census ──────────────────────────────────
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "Computing corpus file census and disk footprint across ${TARGET_DIR}..."

CPP_FILES=$(find "${TARGET_DIR}" \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) 2>/dev/null | wc -l | tr -d ' ')
H_FILES=$(find "${TARGET_DIR}" \( -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \) 2>/dev/null | wc -l | tr -d ' ')
C_FILES=$(find "${TARGET_DIR}" -name "*.c" 2>/dev/null | wc -l | tr -d ' ')
QML_FILES=$(find "${TARGET_DIR}" -name "*.qml" 2>/dev/null | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "${TARGET_DIR}" 2>/dev/null | awk '{print $1}')

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║               Mega-Corpus Harvest Census                       ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s${NC}\n" "Target Directory" "${TARGET_DIR}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s${NC}\n" "Total Disk Footprint" "${TOTAL_SIZE}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "C++ Implementation Files (.cpp/.cc)" "${CPP_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "C/C++ Header Files (.h/.hpp)" "${H_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "Systems C Source Files (.c)" "${C_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "Declarative UI Files (.qml)" "${QML_FILES}"
echo -e "${BOLD}──────────────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}${BOLD}✓ Mega-Corpus harvest completed successfully.${NC}"
echo ""
