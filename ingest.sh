#!/usr/bin/env bash
# =============================================================================
# Titan-LM | Infinite Memory-Flat Streaming Ingestion Pipeline
# Reads source codebases file-by-file, encodes with 4,096 BPE vocabulary,
# and streams raw uint32_t tokens directly into titan_tokens.bin.
# ZERO intermediate text files created. Constant O(1) RAM footprint.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_BIN="${SCRIPT_DIR}/titan_tokens.bin"
VOCAB_FILE="${SCRIPT_DIR}/titan_vocab.txt"
BINARY="${SCRIPT_DIR}/titan_lm"

# ── ANSI Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${CYAN}[ingest]${NC} $*"; }
ok()    { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
die()   { echo -e "${RED}[FATAL ]${NC} $*"; exit 1; }

# ── Sanity Checks ────────────────────────────────────────────────────────────
[[ -x "$BINARY" ]] || die "titan_lm executable not found — run 'make' first"

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     Titan-LM  ·  Infinite Flat-Memory Streaming Pipeline       ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Remove obsolete intermediate text file if present (saves disk space)
if [[ -f "${SCRIPT_DIR}/training_data.txt" ]]; then
    log "Removing obsolete intermediate text file (${SCRIPT_DIR}/training_data.txt)..."
    rm -f "${SCRIPT_DIR}/training_data.txt"
    ok "Intermediate text file removed (disk reclaimed)"
fi

# ── Step 1: Ensure BPE Vocabulary is Ready ───────────────────────────────────
if [[ ! -f "$VOCAB_FILE" ]]; then
    log "titan_vocab.txt not found. Initializing 4,096 BPE vocabulary training..."
    "$BINARY" train-vocab 4096
    ok "Vocabulary initialized and saved to $VOCAB_FILE"
else
    ok "Active 4,096 BPE Vocabulary confirmed: $VOCAB_FILE"
fi

# ── Step 2: Define Target Codebases for Ingestion ─────────────────────────────
# The harvested corpus can live on any volume: set TITAN_CORPUS_DIR for the main
# root and TITAN_LOCAL_PROJECTS (colon-separated) for extra project trees.
CORPUS_DIR="${TITAN_CORPUS_DIR:-${HOME}/titan_massive_dataset}"
TARGET_DIRS=()

if [[ -d "${CORPUS_DIR}" ]]; then
    TARGET_DIRS+=("${CORPUS_DIR}")
fi

# Specialized systems dataset (Folly, Seastar, Abseil, LLaMA.cpp, etc.)
if [[ -d "${HOME}/titan_dataset" ]]; then
    TARGET_DIRS+=("${HOME}/titan_dataset")
fi

# Local project trees — defaults to this repository, override with
# TITAN_LOCAL_PROJECTS="/path/a:/path/b"
LOCAL_PROJECTS="${TITAN_LOCAL_PROJECTS:-${SCRIPT_DIR}}"
IFS=':' read -r -a LOCAL_PROJECT_LIST <<< "${LOCAL_PROJECTS}"
for local_proj in "${LOCAL_PROJECT_LIST[@]}"; do
    if [[ -d "${local_proj}" ]]; then
        TARGET_DIRS+=("${local_proj}")
    fi
done

if (( ${#TARGET_DIRS[@]} == 0 )); then
    die "No corpus roots found. Harvest a corpus with ./harvest_huge_corpus.sh or set TITAN_CORPUS_DIR."
fi

log "Corpus roots to stream (set TITAN_CORPUS_DIR / TITAN_LOCAL_PROJECTS to relocate):"
for d in "${TARGET_DIRS[@]}"; do
    echo -e "  ${CYAN}•${NC} $d ($(du -sh "$d" 2>/dev/null | awk '{print $1}'))"
done
echo ""

# ── Step 3: Run Memory-Flat Streaming Tokenizer Directly to Disk ─────────────
log "Starting streaming tokenization engine directly into ${BOLD}${TOKEN_BIN}${NC}..."
log "Memory Strategy: Flat constant RAM (O(1) file buffer) — direct uint32_t stream"
echo ""

"$BINARY" stream-ingest "$TOKEN_BIN" "${TARGET_DIRS[@]}"

# ── Step 4: Validate Generated Binary Token File ──────────────────────────────
if [[ -f "$TOKEN_BIN" ]]; then
    TOK_SIZE=$(wc -c < "$TOKEN_BIN" | tr -d ' ')
    # Header is 24 bytes, each token is 4 bytes (uint32_t)
    if [[ "$TOK_SIZE" -gt 24 ]]; then
        TOK_COUNT=$(( (TOK_SIZE - 24) / 4 ))
        TOK_MB=$(echo "scale=2; $TOK_SIZE / 1048576" | bc)
        echo ""
        ok "Streaming Ingestion Complete!"
        echo -e "  Contiguous Tokens : ${GREEN}${TOK_COUNT}${NC} tokens"
        echo -e "  Binary File Size  : ${GREEN}${TOK_MB} MB${NC} (${TOK_SIZE} bytes)"
        echo -e "  Target Output     : ${GREEN}${TOKEN_BIN}${NC}"
    else
        die "titan_tokens.bin was created but contains no token payloads"
    fi
else
    die "titan_tokens.bin was not generated — check error logs above"
fi

echo ""
echo -e "${BOLD}── Ready for Training ──────────────────────────────────────────${NC}"
echo -e "  Launch AdamW training loop: ${CYAN}make train${NC}"
echo -e "  Track live in terminal   : ${CYAN}tail -f live_training.log${NC}"
echo ""
