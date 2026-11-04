#!/usr/bin/env bash
# =============================================================================
# Titan-LM | Data Ingestion Pipeline
# Harvests real C++/H/QML source from your projects + stdlib headers,
# merges into training_data.txt, then tokenizes → titan_tokens.bin
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="$SCRIPT_DIR/training_data.txt"
TOKEN_BIN="$SCRIPT_DIR/titan_tokens.bin"
BINARY="$SCRIPT_DIR/titan_lm"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${CYAN}[ingest]${NC} $*"; }
ok()    { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
die()   { echo -e "${RED}[FATAL ]${NC} $*"; exit 1; }

# ── Sanity checks ────────────────────────────────────────────────────────────
[[ -x "$BINARY" ]] || die "titan_lm binary not found — run 'make' first"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        Titan-LM  ·  Data Ingestion Pipeline         ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Define source directories ────────────────────────────────────────
log "Scanning source directories..."

> "$OUT_FILE"   # wipe previous run

TOTAL_FILES=0
TOTAL_BYTES=0

ingest_dir() {
    local label="$1"; shift
    local exts=("$@")
    local count=0

    for dir in \
        "/Users/siluna/Desktop/titan-lm" \
        "/Users/siluna/Desktop/AeroDrop" \
        "/Users/siluna/Desktop/AeroSec" \
        "/Users/siluna/titan_dataset"
    do
        [[ -d "$dir" ]] || continue
        for ext in "${exts[@]}"; do
            while IFS= read -r -d '' f; do
                # Skip build artifacts, git internals, binary objects
                [[ "$f" == */.git/* ]]          && continue
                [[ "$f" == */build/CMakeFiles/* ]] && continue
                [[ "$f" == */.gemini/* ]]        && continue
                [[ "$f" == */node_modules/* ]]   && continue
                # Skip test fixture binary blobs
                [[ "$f" == */testdata/* ]]       && continue

                # Skip empty files
                [[ -s "$f" ]] || continue

                # Inject a clear file boundary comment so the model learns
                # file-level structure (like a document separator)
                printf '\n// === FILE: %s ===\n' "$f" >> "$OUT_FILE"
                cat "$f" >> "$OUT_FILE"
                printf '\n' >> "$OUT_FILE"
                ((count++)) || true
            done < <(find "$dir" -name "$ext" -print0 2>/dev/null)
        done
    done

    TOTAL_FILES=$((TOTAL_FILES + count))
    ok "[$label] $count files ingested"
}

# ── Step 2: Ingest your project source files ─────────────────────────────────
log "Phase A: Ingesting your C++ project files..."
ingest_dir "C++ Sources"  "*.cpp" "*.cc" "*.cxx"
ingest_dir "C++ Headers"  "*.hpp" "*.h"  "*.hxx"
ingest_dir "QML Files"    "*.qml"
ingest_dir "Build Scripts" "Makefile" "*.sh" "CMakeLists.txt"

# ── Step 3: Ingest C++ stdlib headers (massive high-quality C++ corpus) ──────
log "Phase B: Ingesting libc++ stdlib headers (Apple SDK)..."
STDLIB_PATHS=(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX27.2.sdk/usr/include/c++/v1"
    "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/usr/include/c++/v1"
    "/Library/Developer/CommandLineTools/usr/include/c++/v1"
)

STD_COUNT=0
for STDLIB in "${STDLIB_PATHS[@]}"; do
    [[ -d "$STDLIB" ]] || continue
    log "  Found stdlib at: $STDLIB"
    while IFS= read -r -d '' f; do
        [[ -s "$f" ]] || continue
        printf '\n// === STDLIB: %s ===\n' "$(basename "$f")" >> "$OUT_FILE"
        cat "$f" >> "$OUT_FILE"
        printf '\n' >> "$OUT_FILE"
        ((STD_COUNT++)) || true
    done < <(find "$STDLIB" -name "*.h" -print0 2>/dev/null)
    break  # use first valid SDK
done

TOTAL_FILES=$((TOTAL_FILES + STD_COUNT))
ok "[stdlib] $STD_COUNT header files ingested"

# ── Step 4: Stats ────────────────────────────────────────────────────────────
TOTAL_BYTES=$(wc -c < "$OUT_FILE" | tr -d ' ')
TOTAL_LINES=$(wc -l < "$OUT_FILE" | tr -d ' ')
TOTAL_MB=$(echo "scale=2; $TOTAL_BYTES / 1048576" | bc)

echo ""
echo -e "${BOLD}── Ingestion Summary ──────────────────────────────────${NC}"
echo -e "  Source files   : ${GREEN}${TOTAL_FILES}${NC}"
echo -e "  Total lines    : ${GREEN}${TOTAL_LINES}${NC}"
echo -e "  Dataset size   : ${GREEN}${TOTAL_MB} MB${NC} → $OUT_FILE"
echo ""

# ── Step 5: BPE Vocabulary Expansion (Move 2: 4,096 entries) ─────────────────
log "Phase C: Scaling BPE Vocabulary to 4,096 entries..."
echo ""
"$BINARY" train-vocab 4096 "$OUT_FILE"

# ── Step 6: Tokenize ─────────────────────────────────────────────────────────
log "Phase D: Running BPE tokenizer (4,096 vocab) → titan_tokens.bin ..."
echo ""

"$BINARY" tokenize "$OUT_FILE"

if [[ -f "$TOKEN_BIN" ]]; then
    TOK_SIZE=$(wc -c < "$TOKEN_BIN" | tr -d ' ')
    TOK_COUNT=$((TOK_SIZE / 4))   # each token is a uint32_t (4 bytes)
    echo ""
    ok "Tokenization complete!"
    echo -e "  Token count    : ${GREEN}${TOK_COUNT}${NC} tokens"
    echo -e "  Binary size    : ${GREEN}${TOK_SIZE} bytes${NC} → $TOKEN_BIN"
else
    die "titan_tokens.bin was not created — check tokenizer output above"
fi

echo ""
echo -e "${BOLD}── Next Steps ──────────────────────────────────────────${NC}"
echo -e "  1. Run training     : ${CYAN}make train${NC}"
echo -e "  2. Watch loss drop from ~6.8 → below 2.0"
echo -e "  3. Generate code    : ${CYAN}./titan_lm generate \"int main() {\"${NC}"
echo ""
echo -e "${GREEN}${BOLD}✓ Ingestion pipeline complete. Ready to train.${NC}"

