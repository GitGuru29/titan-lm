#!/usr/bin/env bash
# =============================================================================
# Titan-LM | True Turn-Aligned SFT Ingestion Pipeline (Zero-Python C/C++ & General Chat)
# 1. Curated Conversational Greetings & AI Persona
# 2. Pure C/C++ Code Instructions (from iamtarun/code_instructions_120k_alpaca, filtered)
# 3. Clean General Knowledge Q&A (from yahma/alpaca-cleaned, code filtered)
# Streams paired turn-aligned [X, Y_masked] records directly into titan_instruct_tokens.bin.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/titan_lm"
OUTPUT_BIN="${SCRIPT_DIR}/titan_instruct_tokens.bin"
VOCAB_FILE="${SCRIPT_DIR}/titan_vocab.txt"

# ── ANSI Styling ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${CYAN}[ingest-sft]${NC} $*"; }
ok()    { echo -e "${GREEN}[   OK     ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[  WARN    ]${NC} $*"; }
die()   { echo -e "${RED}[  FATAL   ]${NC} $*"; exit 1; }

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Titan-LM · Pure C/C++ & Conversational SFT Ingestion          ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Sanity Checks ────────────────────────────────────────────────────────────
[[ -x "$BINARY" ]] || die "titan_lm binary not found. Run 'make' first."
[[ -f "$VOCAB_FILE" ]] || die "Active vocabulary not found at $VOCAB_FILE. Run 'make' first."

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/titan_sft_XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT
log "Created transient workspace: ${TEMP_DIR}"

GENERAL_PQ="${TEMP_DIR}/general_alpaca.parquet"
CODE_PQ="${TEMP_DIR}/code_120k.parquet"
EXTRACTED_JSONL="${TEMP_DIR}/sft_extracted.jsonl"

# ── Step 1: Download Clean Parquet Splits from Hugging Face ──────────────────
HF_GENERAL_URL="https://huggingface.co/api/datasets/yahma/alpaca-cleaned/parquet/default/train/0.parquet"
HF_CODE_URL="https://huggingface.co/api/datasets/iamtarun/code_instructions_120k_alpaca/parquet/default/train/0.parquet"

log "Downloading general conversational instruction dataset (yahma/alpaca-cleaned)..."
curl -L --retry 3 --connect-timeout 20 -o "$GENERAL_PQ" "$HF_GENERAL_URL" 2>/dev/null
ok "General dataset downloaded: $(du -h "$GENERAL_PQ" | awk '{print $1}')"

log "Downloading high-density code instruction dataset (iamtarun/code_instructions_120k_alpaca)..."
curl -L --retry 3 --connect-timeout 20 -o "$CODE_PQ" "$HF_CODE_URL" 2>/dev/null
ok "Code dataset downloaded: $(du -h "$CODE_PQ" | awk '{print $1}')"

# ── Step 2: Extract Clean C/C++ & Conversational Dialogue ─────────────────────
log "Executing targeted extraction: filtering for 100% pure C/C++ and natural dialogue..."

PYTHONPATH="${SCRIPT_DIR}/build_tools/site-packages:${PYTHONPATH:-}" python3 - << 'EOF' "$TEMP_DIR"
import sys, os, json, duckdb

temp_dir = sys.argv[1]
general_pq = os.path.join(temp_dir, "general_alpaca.parquet")
code_pq = os.path.join(temp_dir, "code_120k.parquet")
out_jsonl = os.path.join(temp_dir, "sft_extracted.jsonl")

total_count = 0
con = duckdb.connect()

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
        ("Who are you?", "I am Titan-LM, an AI assistant built from scratch in C++20 for Apple Silicon. I can answer questions, explain concepts, and write high-performance C and C++ code."),
        ("What is your name?", "I am Titan-LM, a bare-metal Transformer AI model developed in modern C++20."),
        ("What can you do?", "I can answer general questions, explain technical concepts, write and debug software code (especially C++ and systems programming), and engage in conversation."),
        ("How are you?", "I am doing well, thank you! How can I help you today?"),
        ("How are you doing?", "I'm doing great and ready to assist you. What are you working on?"),
        ("Thank you", "You're very welcome! Let me know if you need anything else."),
        ("Thanks", "Glad I could help! Feel free to ask if you have more questions."),
        ("Bye", "Goodbye! Have a great day!"),
        ("Goodbye", "Goodbye! Feel free to reach out whenever you need assistance."),
        ("What are the primary colors?", "The primary colors of light are red, green, and blue (RGB). In traditional art and pigments, the primary colors are red, yellow, and blue."),
        ("Describe an atom.", "An atom is the basic unit of a chemical element, consisting of a central nucleus made of protons and neutrons, surrounded by a cloud of electrons."),
        ("Why is the sky blue?", "The sky appears blue because molecules in Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves (Rayleigh scattering).")
    ]
    # Replicate persona dialogues with high frequency so greetings have solid probability mass
    for _ in range(100):
        for q, a in persona_dialogues:
            out.write(json.dumps({"instruction": q, "output": a}, ensure_ascii=False) + "\n")
            total_count += 1
    print(f"[extractor] Injected {len(persona_dialogues) * 100} reinforced conversational greetings and persona turns.")

    # 2. Extract PURE C & C++ Coding Instructions (Strictly exclude Python!)
    if os.path.exists(code_pq):
        q_cpp = f"""
        SELECT instruction, input, output FROM '{code_pq}'
        WHERE (
            LOWER(instruction) LIKE '%c++%' OR LOWER(instruction) LIKE '% c %' 
            OR LOWER(instruction) LIKE '%cpp%' OR LOWER(output) LIKE '%#include%' 
            OR LOWER(output) LIKE '%std::%' OR LOWER(output) LIKE '%int main(%'
        )
        AND LOWER(output) NOT LIKE '%def %'
        AND LOWER(output) NOT LIKE '%import %'
        AND LOWER(output) NOT LIKE '%print(%'
        AND LOWER(instruction) NOT LIKE '%python%'
        """
        cpp_rows = con.execute(q_cpp).fetchall()
        cpp_count = 0
        for inst, inp, resp in cpp_rows:
            inst = (inst or "").strip()
            inp = (inp or "").strip()
            resp = (resp or "").strip()
            if inp:
                inst = f"{inst}\n\nContext:\n{inp}"
            if inst and resp:
                out.write(json.dumps({"instruction": inst, "output": resp}, ensure_ascii=False) + "\n")
                cpp_count += 1
        print(f"[extractor] Extracted {cpp_count} pure C/C++ programming pairs (zero Python).")
        total_count += cpp_count

    # 3. Extract General Knowledge & Explanations from Alpaca Cleaned (Exclude coding/python)
    if os.path.exists(general_pq):
        q_gen = f"""
        SELECT instruction, input, output FROM '{general_pq}'
        WHERE LOWER(output) NOT LIKE '%def %'
          AND LOWER(output) NOT LIKE '%import %'
          AND LOWER(output) NOT LIKE '%print(%'
          AND LOWER(instruction) NOT LIKE '%python%'
          AND LOWER(instruction) NOT LIKE '%code%'
          AND LENGTH(instruction) > 10
          AND LENGTH(output) > 10
        LIMIT 20000
        """
        gen_rows = con.execute(q_gen).fetchall()
        gen_count = 0
        for inst, inp, resp in gen_rows:
            inst = (inst or "").strip()
            inp = (inp or "").strip()
            resp = (resp or "").strip()
            if inp:
                inst = f"{inst}\n\nContext:\n{inp}"
            if inst and resp:
                out.write(json.dumps({"instruction": inst, "output": resp}, ensure_ascii=False) + "\n")
                gen_count += 1
        print(f"[extractor] Extracted {gen_count} clean general knowledge & dialogue pairs.")
        total_count += gen_count

print(f"[extractor] Total SFT training pairs compiled: {total_count}")
EOF

ok "Extraction complete: $(wc -l < "$EXTRACTED_JSONL" | tr -d ' ') paired conversation rows ready"

# ── Step 3: Stream Aligned (X, Y_masked) Records to Binary ───────────────────
log "Streaming turn-aligned SFT records (with prompt loss masking) to ${BOLD}${OUTPUT_BIN}${NC}..."
"$BINARY" stream-instruct "$EXTRACTED_JSONL" "$OUTPUT_BIN" 192

# ── Step 4: Verification ─────────────────────────────────────────────────────
if [[ -f "$OUTPUT_BIN" ]]; then
    TOK_SIZE=$(wc -c < "$OUTPUT_BIN" | tr -d ' ')
    if [[ "$TOK_SIZE" -gt 24 ]]; then
        TOK_COUNT=$(( (TOK_SIZE - 24) / 4 ))
        RECORD_STRIDE=$(( 2 * 192 ))
        TOTAL_RECORDS=$(( TOK_COUNT / RECORD_STRIDE ))
        TOK_MB=$(echo "scale=2; $TOK_SIZE / 1048576" | bc)
        echo ""
        ok "Turn-Aligned SFT Ingestion Complete!"
        echo -e "  Total Aligned Records : ${GREEN}${TOTAL_RECORDS}${NC} complete dialogue turns"
        echo -e "  Record Stride         : ${GREEN}${RECORD_STRIDE}${NC} tokens (192 input X + 192 masked target Y)"
        echo -e "  Binary File Size      : ${GREEN}${TOK_MB} MB${NC} (${TOK_SIZE} bytes)"
        echo -e "  Target Output         : ${GREEN}${OUTPUT_BIN}${NC}"
        echo ""
        echo -e "${BOLD}── Ready for Turn-Aligned SFT Alignment ────────────────────────${NC}"
        echo -e "  Launch Alignment Loop : ${CYAN}make train-instruct${NC}  (or: ./titan_lm train_instruct)"
        echo -e "  Track Live Telemetry  : ${CYAN}tail -f live_training.log${NC}"
        echo ""
    else
        die "Output binary contains no records"
    fi
else
    die "Failed to generate $OUTPUT_BIN"
fi
