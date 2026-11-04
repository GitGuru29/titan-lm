#!/usr/bin/env bash
# =============================================================================
# Titan-LM | Production Mega-Corpus Harvester
# Shallow-clones Tier-1 C/C++/Systems codebases into a configurable corpus root.
#
# Usage:
#   ./harvest_huge_corpus.sh                          # default: ~/titan_massive_dataset
#   ./harvest_huge_corpus.sh /Volumes/SSD/titan_corpus
#   TITAN_CORPUS_DIR=/Volumes/SSD/titan_corpus ./harvest_huge_corpus.sh
#   TITAN_JOBS=8 ./harvest_huge_corpus.sh             # parallel clone workers
# =============================================================================

set -euo pipefail

# ── Configuration (env-overridable) ──────────────────────────────────────────
CORPUS_DIR="${1:-${TITAN_CORPUS_DIR:-${HOME}/titan_massive_dataset}}"
JOBS="${TITAN_JOBS:-4}"
MIN_FREE_GB="${TITAN_MIN_FREE_GB:-10}"

PARTIAL_DIR="${CORPUS_DIR}/.partial"
LOG_DIR="${CORPUS_DIR}/.harvest_logs"
RESULTS_FILE="${LOG_DIR}/results.tsv"

# A corpus is only useful once it exceeds a few GB; refuse to fill the disk.
if (( JOBS < 1 )); then JOBS=1; fi

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
echo -e "${BOLD}║      30+ Tier-1 Systems, OS, Compiler & Performance Codebases  ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Initialize Dedicated Target Directory ────────────────────────────
log "Corpus root: ${BOLD}${CORPUS_DIR}${NC}"
mkdir -p "${CORPUS_DIR}" "${PARTIAL_DIR}" "${LOG_DIR}"
cd "${CORPUS_DIR}"
CORPUS_DIR="$(pwd)"
PARTIAL_DIR="${CORPUS_DIR}/.partial"
LOG_DIR="${CORPUS_DIR}/.harvest_logs"
RESULTS_FILE="${LOG_DIR}/results.tsv"
: > "${RESULTS_FILE}"
ok "Corpus root ready: $(pwd)"

# Free-space guard: a full harvest of these repos needs tens of GB.
AVAIL_GB=$(df -Pk "${CORPUS_DIR}" | awk 'NR==2 {printf "%.1f", $4 / 1048576}')
log "Available disk space: ${AVAIL_GB} GB"
if awk -v a="${AVAIL_GB}" -v m="${MIN_FREE_GB}" 'BEGIN { exit !(a < m) }'; then
    warn "Only ${AVAIL_GB} GB free (< ${MIN_FREE_GB} GB). Point TITAN_CORPUS_DIR at a larger volume"
    warn "or lower the threshold with TITAN_MIN_FREE_GB to continue anyway."
fi
echo ""

# ── Helper: Robust Atomic Shallow Clone (runs in worker processes) ───────────
# Clones into a staging directory and renames it into place only on success, so
# an interrupted run can never leave a half-populated repo that is skipped later.
harvest_one() {
    local name="$1"
    local url="$2"
    local dest="${CORPUS_DIR}/${name}"
    local staging="${PARTIAL_DIR}/${name}"
    local repo_log="${LOG_DIR}/${name}.log"
    local status size elapsed t0 t1 attempt

    t0=$(date +%s)

    if [[ -d "${dest}/.git" || (-d "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null)") ]]; then
        status="SKIP"
    else
        : > "${repo_log}"
        status="FAIL"
        attempt=1
        while (( attempt <= 2 )); do
            rm -rf "${staging}"
            # --depth 1 already fetches only tip blobs; adding --filter=blob:none
            # would force a second on-demand fetch during checkout.
            if GIT_TERMINAL_PROMPT=0 git clone --quiet --depth 1 --single-branch --no-tags \
                    "${url}" "${staging}" >>"${repo_log}" 2>&1; then
                mv "${staging}" "${dest}"
                status="OK"
                break
            fi
            echo "--- clone attempt ${attempt} failed ---" >>"${repo_log}"
            attempt=$((attempt + 1))
        done
        rm -rf "${staging}"
    fi

    t1=$(date +%s)
    elapsed=$((t1 - t0))
    size=$([[ -d "${dest}" ]] && du -sh "${dest}" 2>/dev/null | awk '{print $1}' || echo "-")

    printf '%s\t%s\t%s\t%s\n' "${status}" "${name}" "${size}" "${elapsed}" >>"${RESULTS_FILE}"

    case "${status}" in
        OK)   printf '%b[ OK ]%b %-22s %7s  %4ss\n' "${GREEN}"  "${NC}" "${name}" "${size}" "${elapsed}" ;;
        SKIP) printf '%b[SKIP]%b %-22s %7s  %4ss  already present\n' "${YELLOW}" "${NC}" "${name}" "${size}" "${elapsed}" ;;
        FAIL) printf '%b[FAIL]%b %-22s %7s  %4ss  log: %s\n' "${RED}" "${NC}" "${name}" "${size}" "${elapsed}" "${repo_log}" ;;
    esac
}
export -f harvest_one
export CORPUS_DIR PARTIAL_DIR LOG_DIR RESULTS_FILE GREEN YELLOW RED NC

# ── Step 2: Tier-1 Systems & Engineering Codebases ───────────────────────────
repo_list() {
    cat <<'REPOS'
linux|https://github.com/torvalds/linux.git|Linux Kernel - Definitive standard for systems C, drivers, memory management & low-level concurrency
freebsd-src|https://github.com/freebsd/freebsd-src.git|FreeBSD - Production BSD Unix kernel, ZFS integration, networking stack, libc & userland
serenity|https://github.com/SerenityOS/serenity.git|SerenityOS - Modern, bespoke C++20 Unix-like operating system kernel, GUI & libraries
haiku|https://github.com/haiku/haiku.git|Haiku OS - BeOS-compatible C++ desktop OS with modular microkernel & object-oriented API
reactos|https://github.com/reactos/reactos.git|ReactOS - Open-source Windows NT architecture reimplementation in systems C & C++
llvm-project|https://github.com/llvm/llvm-project.git|LLVM/Clang/MLIR - State of the art compiler infrastructure & template-heavy modern C++20
gcc|https://github.com/gcc-mirror/gcc.git|GNU Compiler Collection - Historic & definitive C/C++ compiler and runtime libraries
tree-sitter|https://github.com/tree-sitter/tree-sitter.git|Tree-sitter - High-speed incremental AST parser generator in C/C++
mold|https://github.com/rui314/mold.git|Mold - Blazing-fast modern parallel C++20 ELF/Mach-O linker
CMake|https://github.com/Kitware/CMake.git|CMake - Definitive cross-platform build system generator in C++
mimalloc|https://github.com/microsoft/mimalloc.git|mimalloc - Compact high-performance secure general-purpose memory allocator in C
jemalloc|https://github.com/jemalloc/jemalloc.git|jemalloc - Fragmentation-resistant multi-threaded memory allocator
folly|https://github.com/facebook/folly.git|Facebook Folly - High-performance modern C++20 library with fibers, async IO & atomics
abseil-cpp|https://github.com/abseil/abseil-cpp.git|Abseil C++ - Google's standard C++20 library extensions (SwissTable, synchronization, strings)
taskflow|https://github.com/taskflow/taskflow.git|Taskflow - Modern C++ parallel & heterogeneous task programming graph engine
oneTBB|https://github.com/oneapi-src/oneTBB.git|oneTBB - Intel Threading Building Blocks for multi-core concurrent parallelism
concurrentqueue|https://github.com/cameron314/concurrentqueue.git|concurrentqueue - Blazing-fast lock-free multi-producer multi-consumer queue in C++
envoy|https://github.com/envoyproxy/envoy.git|Envoy - Cloud-native high-performance edge/service proxy in modern C++
libuv|https://github.com/libuv/libuv.git|libuv - High-performance asynchronous I/O event loop powering Node.js
grpc|https://github.com/grpc/grpc.git|gRPC - Universal high-performance RPC framework in C & C++
rocksdb|https://github.com/facebook/rocksdb.git|RocksDB - High-performance LSM-tree embedded key-value storage engine in C++
duckdb|https://github.com/duckdb/duckdb.git|DuckDB - State of the art vectorized analytical columnar database management system
postgres|https://github.com/postgres/postgres.git|PostgreSQL - The world's most advanced relational database engine in pure systems C
zfs|https://github.com/openzfs/zfs.git|OpenZFS - Enterprise-grade advanced copy-on-write file system & logical volume manager
Hyprland|https://github.com/hyprwm/Hyprland.git|Hyprland - Dynamic tiling Wayland compositor in modern C++23
qtbase|https://github.com/qt/qtbase.git|QtBase - Core C++ libraries of the Qt framework (QtCore, QtGui, QtWidgets, QtNetwork)
FFmpeg|https://github.com/FFmpeg/FFmpeg.git|FFmpeg - Industry-standard multimedia codec, multiplexer, and filter engine in C
openssl|https://github.com/openssl/openssl.git|OpenSSL - Cryptographic algorithms, TLS/SSL protocols, and security primitives
libbpf|https://github.com/libbpf/libbpf.git|libbpf - Linux eBPF user-space loader and helper library for kernel observability
llama.cpp|https://github.com/ggml-org/llama.cpp.git|llama.cpp - High-performance bare-metal LLM inference and quantized kernels in C/C++
opencv|https://github.com/opencv/opencv.git|OpenCV - High-performance numerical linear algebra, image matrices, and SIMD kernels
tensorflow|https://github.com/tensorflow/tensorflow.git|TensorFlow - Production C++ compute graphs, autograd operations, and tensor engines
REPOS
}

REPO_COUNT=$(repo_list | grep -c '|' || true)

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "Harvesting ${BOLD}${REPO_COUNT}${NC} codebases | workers=${JOBS} | depth=1 | atomic staging + 1 retry"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Clones run through xargs so a large repo cannot monopolise the run and one
# failed repository no longer aborts the remaining harvest.
repo_list | while IFS='|' read -r name url desc; do
    [[ -z "${name}" ]] && continue
    printf '%s\0%s\0%s\0' "${name}" "${url}" "${desc}"
done | xargs -0 -n 3 -P "${JOBS}" bash -c 'harvest_one "$@"' _

echo ""

# ── Step 3: Dataset Telemetry & File Census ──────────────────────────────────
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log "Computing corpus file census and disk footprint across ${CORPUS_DIR}..."

# Single traversal of the tree; the old version walked the corpus five times.
CPP_FILES=0
H_FILES=0
C_FILES=0
QML_FILES=0
while IFS= read -r -d '' file; do
    case "${file}" in
        *.cpp|*.cc|*.cxx) CPP_FILES=$((CPP_FILES + 1)) ;;
        *.h|*.hpp|*.hxx)  H_FILES=$((H_FILES + 1)) ;;
        *.c)              C_FILES=$((C_FILES + 1)) ;;
        *.qml)            QML_FILES=$((QML_FILES + 1)) ;;
    esac
done < <(find "${CORPUS_DIR}" -path "${PARTIAL_DIR}" -prune -o -type f -print0 2>/dev/null)

TOTAL_SIZE=$(du -sh "${CORPUS_DIR}" 2>/dev/null | awk '{print $1}')

CLONED=$(awk -F'\t' '$1 == "OK"   { n++ } END { print n + 0 }' "${RESULTS_FILE}")
SKIPPED=$(awk -F'\t' '$1 == "SKIP" { n++ } END { print n + 0 }' "${RESULTS_FILE}")
FAILED=$(awk -F'\t' '$1 == "FAIL" { n++ } END { print n + 0 }' "${RESULTS_FILE}")

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║               Mega-Corpus Harvest Census                       ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s${NC}\n" "Corpus Root" "${CORPUS_DIR}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s${NC}\n" "Total Disk Footprint" "${TOTAL_SIZE}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "C++ Implementation Files (.cpp/.cc)" "${CPP_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "C/C++ Header Files (.h/.hpp)" "${H_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "Systems C Source Files (.c)" "${C_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s files${NC}\n" "Declarative UI Files (.qml)" "${QML_FILES}"
printf "  ${CYAN}%-32s${NC} : ${GREEN}%s new · %s existing · %s failed${NC}\n" "Repositories" "${CLONED}" "${SKIPPED}" "${FAILED}"
echo -e "${BOLD}──────────────────────────────────────────────────────────────────${NC}"

if (( FAILED > 0 )); then
    warn "${FAILED} repository clone(s) failed. Re-run to retry them; per-repo logs: ${LOG_DIR}"
    echo -e "${YELLOW}${BOLD}! Mega-Corpus harvest completed with gaps.${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}${BOLD}✓ Mega-Corpus harvest completed successfully.${NC}"
echo -e "  Next: point the tokenizer at it and stream tokens:"
echo -e "    ${CYAN}TITAN_CORPUS_DIR=${CORPUS_DIR} make ingest${NC}"
echo ""
