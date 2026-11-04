#!/usr/bin/env bash
# =============================================================================
# Titan-LM | Phase 9: Conversational Alignment & Instruction Ingestion Pipeline
# Downloads verified clean Hugging Face instruction corpus split (Parquet),
# extracts {instruction, output} conversational pairs via lightweight helper,
# wraps in strict chat format "<|user|>\n{instruction}\n<|assistant|>\n{output}<|end_of_text|>",
# and streams uint32_t tokens straight to titan_instruct_tokens.bin with flat O(1) RAM.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/titan_lm"
OUTPUT_BIN="${SCRIPT_DIR}/titan_instruct_tokens.bin"
VOCAB_FILE="${SCRIPT_DIR}/titan_vocab.txt"

# ── ANSI Styling ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${CYAN}[ingest-instruct]${NC} $*"; }
ok()    { echo -e "${GREEN}[      OK       ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[     WARN      ]${NC} $*"; }
die()   { echo -e "${RED}[    FATAL      ]${NC} $*"; exit 1; }

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Titan-LM · Phase 9 Conversational Instruction Ingestion       ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Sanity Checks ────────────────────────────────────────────────────────────
[[ -x "$BINARY" ]] || die "titan_lm binary not found. Run 'make' first."
[[ -f "$VOCAB_FILE" ]] || die "Active vocabulary not found at $VOCAB_FILE. Run 'make' first."

# ── Step 1: Create Temporary Ingestion Workspace ─────────────────────────────
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/titan_instruct_XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT
log "Created transient workspace: ${TEMP_DIR}"

PARQUET_FILE="${TEMP_DIR}/code_instruct.parquet"
EXTRACTED_JSONL="${TEMP_DIR}/extracted.jsonl"

# ── Step 2: Download Verified Instruction Corpus Split from Hugging Face ─────
# Primary target: Sahil2801 / CodeAlpaca-20k (20,022 clean code instruction pairs)
HF_PARQUET_URL="https://huggingface.co/api/datasets/sahil2801/CodeAlpaca-20k/parquet/default/train/0.parquet"
HF_JSON_FALLBACK_URL="https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json"

log "Downloading verified instruction split from Hugging Face Hub..."
log "Target: ${HF_PARQUET_URL}"

DOWNLOAD_SUCCESS=false
if curl -L --retry 3 --connect-timeout 15 -o "$PARQUET_FILE" "$HF_PARQUET_URL" 2>/dev/null; then
    # Verify valid Parquet magic bytes ('PAR1')
    if [[ -f "$PARQUET_FILE" ]] && [[ $(head -c 4 "$PARQUET_FILE") == "PAR1" ]]; then
        DOWNLOAD_SUCCESS=true
        ok "Parquet split downloaded successfully: $(du -h "$PARQUET_FILE" | awk '{print $1}')"
    fi
fi

if [[ "$DOWNLOAD_SUCCESS" != "true" ]]; then
    warn "Parquet download direct stream failed. Fetching verified JSON split fallback..."
    JSON_FILE="${TEMP_DIR}/code_instruct.json"
    curl -L --retry 3 --connect-timeout 15 -o "$JSON_FILE" "$HF_JSON_FALLBACK_URL"
    ok "JSON split downloaded successfully: $(du -h "$JSON_FILE" | awk '{print $1}')"
fi

# ── Step 3: Lightweight Extraction Utility ───────────────────────────────────
# Extracts instruction and output objects without installing massive ML packages
log "Executing lightweight extraction routine into JSON lines..."

PYTHONPATH="${SCRIPT_DIR}/build_tools/site-packages:${PYTHONPATH:-}" python3 - << 'EOF' "$TEMP_DIR"
import sys, os, json

temp_dir = sys.argv[1]
parquet_path = os.path.join(temp_dir, "code_instruct.parquet")
json_path = os.path.join(temp_dir, "code_instruct.json")
out_jsonl = os.path.join(temp_dir, "extracted.jsonl")

extracted_count = 0

def write_rows(rows, out_file):
    count = 0
    with open(out_file, "w", encoding="utf-8") as out:
        for r in rows:
            inst = (r.get("instruction") or r.get("prompt") or "").strip()
            inp = (r.get("input") or "").strip()
            resp = (r.get("output") or r.get("response") or "").strip()
            if inp:
                inst = f"{inst}\n\nContext:\n{inp}"
            if inst and resp:
                out.write(json.dumps({"instruction": inst, "output": resp}, ensure_ascii=False) + "\n")
                count += 1
    return count

# Method A: Parquet extraction via lightweight DuckDB or PyArrow
if os.path.exists(parquet_path):
    try:
        import duckdb
        conn = duckdb.connect()
        query = f"SELECT instruction, input, output FROM '{parquet_path}'"
        cursor = conn.execute(query)
        with open(out_jsonl, "w", encoding="utf-8") as out:
            for inst, inp, resp in cursor.fetchall():
                inst = (inst or "").strip()
                inp = (inp or "").strip()
                resp = (resp or "").strip()
                if inp:
                    inst = f"{inst}\n\nContext:\n{inp}"
                if inst and resp:
                    out.write(json.dumps({"instruction": inst, "output": resp}, ensure_ascii=False) + "\n")
                    extracted_count += 1
        print(f"[extractor] DuckDB extracted {extracted_count} dialogue pairs from Parquet split.")
    except Exception as e_duck:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            extracted_count = write_rows(df.to_dict(orient="records"), out_jsonl)
            print(f"[extractor] PyArrow extracted {extracted_count} dialogue pairs from Parquet split.")
        except Exception as e_arrow:
            print(f"[extractor] Parquet engine unavailable ({e_duck}, {e_arrow}). Switching to JSON parsing...")

# Method B: Pure Python Standard Library fallback (0 external dependencies)
if extracted_count == 0 and os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)
        extracted_count = write_rows(data, out_jsonl)
        print(f"[extractor] Standard JSON parsed {extracted_count} dialogue pairs.")

if extracted_count == 0:
    print("[ERROR] Failed to extract dialogue items from downloaded dataset.", file=sys.stderr)
    sys.exit(1)
EOF

ok "Extraction complete: $(wc -l < "$EXTRACTED_JSONL" | tr -d ' ') conversation rows generated"

# ── Step 4: Stream Tokens Directly to titan_instruct_tokens.bin ─────────────
log "Streaming structured chat tokens directly into ${BOLD}${OUTPUT_BIN}${NC}..."
log "Memory strategy: Flat constant O(1) buffer — zero monolithic allocations"
echo ""

"$BINARY" stream-instruct "$EXTRACTED_JSONL" "$OUTPUT_BIN"

# ── Step 5: Verify Generated Binary File ─────────────────────────────────────
if [[ -f "$OUTPUT_BIN" ]]; then
    TOK_SIZE=$(wc -c < "$OUTPUT_BIN" | tr -d ' ')
    if [[ "$TOK_SIZE" -gt 24 ]]; then
        TOK_COUNT=$(( (TOK_SIZE - 24) / 4 ))
        TOK_MB=$(echo "scale=2; $TOK_SIZE / 1048576" | bc)
        echo ""
        ok "Phase 9 Conversational Ingestion Complete!"
        echo -e "  Contiguous Tokens : ${GREEN}${TOK_COUNT}${NC} tokens"
        echo -e "  Binary File Size  : ${GREEN}${TOK_MB} MB${NC} (${TOK_SIZE} bytes)"
        echo -e "  Target Output     : ${GREEN}${OUTPUT_BIN}${NC}"
        echo ""
        echo -e "${BOLD}── Ready for Instruction Fine-Tuning ──────────────────────────${NC}"
        echo -e "  Launch Alignment Loop : ${CYAN}make train-instruct${NC}  (or: ./titan_lm train_instruct)"
        echo -e "  Track Live Telemetry  : ${CYAN}tail -f live_training.log${NC}"
        echo ""
    else
        die "Output binary was created but contains no token payloads"
    fi
else
    die "Failed to generate $OUTPUT_BIN"
fi

