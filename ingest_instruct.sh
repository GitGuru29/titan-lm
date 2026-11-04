#!/usr/bin/env bash
# =============================================================================
# Titan-LM | Conversational Alignment & General Instruction Ingestion Pipeline
# Downloads:
#   1. yahma/alpaca-cleaned (51,760 general AI chat, QA, and reasoning pairs)
#   2. sahil2801/CodeAlpaca-20k (20,016 programming & algorithmic pairs)
#   3. Curated conversational greetings and AI persona dialogues
# Wraps all pairs in strict chat template:
#   "<|user|>\n{instruction}\n<|assistant|>\n{output}<|end_of_text|>"
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
echo -e "${BOLD}║  Titan-LM · General Chat & Coding Alignment Ingestion         ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Sanity Checks ────────────────────────────────────────────────────────────
[[ -x "$BINARY" ]] || die "titan_lm binary not found. Run 'make' first."
[[ -f "$VOCAB_FILE" ]] || die "Active vocabulary not found at $VOCAB_FILE. Run 'make' first."

# ── Step 1: Create Temporary Ingestion Workspace ─────────────────────────────
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/titan_instruct_XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT
log "Created transient workspace: ${TEMP_DIR}"

GENERAL_PARQUET="${TEMP_DIR}/general_alpaca.parquet"
CODE_PARQUET="${TEMP_DIR}/code_alpaca.parquet"
EXTRACTED_JSONL="${TEMP_DIR}/extracted.jsonl"

# ── Step 2: Download General Chat + Coding Instruction Splits ────────────────
# 1. General AI & Conversational QA: yahma/alpaca-cleaned (51.7k clean entries)
HF_GENERAL_URL="https://huggingface.co/api/datasets/yahma/alpaca-cleaned/parquet/default/train/0.parquet"
# 2. Coding & Technical Implementation: sahil2801/CodeAlpaca-20k (20k clean code entries)
HF_CODE_URL="https://huggingface.co/api/datasets/sahil2801/CodeAlpaca-20k/parquet/default/train/0.parquet"

log "Downloading general conversational instruction dataset (Alpaca Cleaned)..."
if curl -L --retry 3 --connect-timeout 20 -o "$GENERAL_PARQUET" "$HF_GENERAL_URL" 2>/dev/null; then
    ok "General instruction dataset downloaded: $(du -h "$GENERAL_PARQUET" | awk '{print $1}')"
else
    warn "General instruction parquet download failed."
fi

log "Downloading high-density code instruction dataset (CodeAlpaca-20k)..."
if curl -L --retry 3 --connect-timeout 20 -o "$CODE_PARQUET" "$HF_CODE_URL" 2>/dev/null; then
    ok "Coding instruction dataset downloaded: $(du -h "$CODE_PARQUET" | awk '{print $1}')"
else
    warn "Code instruction parquet download failed."
fi

# ── Step 3: Extract & Synthesize Balanced Conversational JSON Lines ───────────
log "Extracting and synthesizing general chat, greetings, and coding pairs..."

PYTHONPATH="${SCRIPT_DIR}/build_tools/site-packages:${PYTHONPATH:-}" python3 - << 'EOF' "$TEMP_DIR"
import sys, os, json

temp_dir = sys.argv[1]
general_pq = os.path.join(temp_dir, "general_alpaca.parquet")
code_pq = os.path.join(temp_dir, "code_alpaca.parquet")
out_jsonl = os.path.join(temp_dir, "extracted.jsonl")

total_count = 0

with open(out_jsonl, "w", encoding="utf-8") as out:
    # 1. Curated Conversational Greetings & Persona Dialogues (High frequency reinforcement)
    persona_dialogues = [
        ("Hi", "Hello! How can I help you today?"),
        ("Hello", "Hi there! What can I assist you with?"),
        ("Hey", "Hello! How can I assist you today?"),
        ("Hi there", "Hello! I am ready to help. What's on your mind?"),
        ("Good morning", "Good morning! How can I help you today?"),
        ("Good afternoon", "Good afternoon! How can I assist you?"),
        ("Good evening", "Good evening! What can I help you with today?"),
        ("Who are you?", "I am Titan-LM, an AI assistant built from scratch in C++20 for Apple Silicon. I can help answer questions, explain concepts, and write code."),
        ("What is your name?", "I am Titan-LM, a bare-metal Transformer AI model developed in modern C++20."),
        ("What can you do?", "I can answer general questions, explain complex ideas, write and debug software code (especially C++ and systems programming), and engage in conversation."),
        ("How are you?", "I am doing well, thank you! How can I help you today?"),
        ("How are you doing?", "I'm doing great and ready to assist you. What are you working on?"),
        ("Thank you", "You're very welcome! Let me know if you need anything else."),
        ("Thanks", "Glad I could help! Feel free to ask if you have more questions."),
        ("Bye", "Goodbye! Have a great day!"),
        ("Goodbye", "Goodbye! Feel free to reach out whenever you need assistance.")
    ]
    # Replicate persona dialogues to ensure strong probability mass during alignment
    for _ in range(50):
        for q, a in persona_dialogues:
            out.write(json.dumps({"instruction": q, "output": a}, ensure_ascii=False) + "\n")
            total_count += 1

    # Helper to extract from parquet via duckdb or pyarrow
    def extract_parquet(path, label):
        global total_count
        if not os.path.exists(path):
            return
        try:
            import duckdb
            con = duckdb.connect()
            cursor = con.execute(f"SELECT instruction, input, output FROM '{path}'")
            count = 0
            for inst, inp, resp in cursor.fetchall():
                inst = (inst or "").strip()
                inp = (inp or "").strip()
                resp = (resp or "").strip()
                if inp:
                    inst = f"{inst}\n\nContext:\n{inp}"
                if inst and resp:
                    out.write(json.dumps({"instruction": inst, "output": resp}, ensure_ascii=False) + "\n")
                    count += 1
            print(f"[extractor] DuckDB extracted {count} items from {label}.")
            total_count += count
        except Exception as e:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                df = table.to_pandas()
                count = 0
                for r in df.to_dict(orient="records"):
                    inst = (r.get("instruction") or "").strip()
                    inp = (r.get("input") or "").strip()
                    resp = (r.get("output") or "").strip()
                    if inp:
                        inst = f"{inst}\n\nContext:\n{inp}"
                    if inst and resp:
                        out.write(json.dumps({"instruction": inst, "output": resp}, ensure_ascii=False) + "\n")
                        count += 1
                print(f"[extractor] PyArrow extracted {count} items from {label}.")
                total_count += count
            except Exception as e2:
                print(f"[extractor] Error extracting {label}: {e}, {e2}")

    extract_parquet(general_pq, "General Alpaca (Conversational QA)")
    extract_parquet(code_pq, "CodeAlpaca (Programming & Algorithms)")

print(f"[extractor] Successfully compiled {total_count} total conversational training pairs.")
if total_count == 0:
    sys.exit(1)
EOF

ok "Dataset extraction complete: $(wc -l < "$EXTRACTED_JSONL" | tr -d ' ') rows prepared"

# ── Step 4: Stream Tokens Directly to titan_instruct_tokens.bin ─────────────
log "Streaming structured chat tokens directly into ${BOLD}${OUTPUT_BIN}${NC}..."
log "Memory strategy: Flat constant O(1) buffer — zero intermediate text files"
echo ""

"$BINARY" stream-instruct "$EXTRACTED_JSONL" "$OUTPUT_BIN"

# ── Step 5: Verify Generated Binary File ─────────────────────────────────────
if [[ -f "$OUTPUT_BIN" ]]; then
    TOK_SIZE=$(wc -c < "$OUTPUT_BIN" | tr -d ' ')
    if [[ "$TOK_SIZE" -gt 24 ]]; then
        TOK_COUNT=$(( (TOK_SIZE - 24) / 4 ))
        TOK_MB=$(echo "scale=2; $TOK_SIZE / 1048576" | bc)
        echo ""
        ok "General Chat & Coding Ingestion Complete!"
        echo -e "  Contiguous Tokens : ${GREEN}${TOK_COUNT}${NC} tokens"
        echo -e "  Binary File Size  : ${GREEN}${TOK_MB} MB${NC} (${TOK_SIZE} bytes)"
        echo -e "  Target Output     : ${GREEN}${OUTPUT_BIN}${NC}"
        echo ""
        echo -e "${BOLD}── Ready for Conversational Instruction Alignment ─────────────${NC}"
        echo -e "  Launch Alignment Loop : ${CYAN}make train-instruct${NC}  (or: ./titan_lm train_instruct)"
        echo -e "  Track Live Telemetry  : ${CYAN}tail -f live_training.log${NC}"
        echo ""
    else
        die "Output binary was created but contains no token payloads"
    fi
else
    die "Failed to generate $OUTPUT_BIN"
fi
