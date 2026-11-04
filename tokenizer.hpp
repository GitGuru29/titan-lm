#pragma once
#ifndef TITAN_TOKENIZER_HPP
#define TITAN_TOKENIZER_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <span>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <optional>
#include <chrono>

namespace titan {

// Flat contiguous integer token representation (32-bit for large vocabularies & modern LLM scaling)
using TokenId = uint32_t;

// Canonical special token identifiers
constexpr TokenId PAD_TOKEN_ID         = 0;
constexpr TokenId BOS_TOKEN_ID         = 1;
constexpr TokenId EOS_TOKEN_ID         = 2;
constexpr TokenId UNK_TOKEN_ID         = 3;
constexpr TokenId SPECIAL_TOKENS_COUNT = 4;

// Base byte tokens: bytes [0x00, 0xFF] are mapped to IDs [4, 259]
constexpr TokenId BYTE_TOKEN_OFFSET    = 4;
constexpr TokenId BASE_VOCAB_SIZE      = SPECIAL_TOKENS_COUNT + 256; // 260

// String representations for canonical special tokens
constexpr std::string_view PAD_TOKEN_STR = "<|pad|>";
constexpr std::string_view BOS_TOKEN_STR = "<|begin_of_text|>";
constexpr std::string_view EOS_TOKEN_STR = "<|end_of_text|>";
constexpr std::string_view UNK_TOKEN_STR = "<|unk|>";

// High-performance 64-bit pair hash using SplitMix64 avalanche mixer
struct PairHash {
    [[nodiscard]] std::size_t operator()(const std::pair<TokenId, TokenId>& p) const noexcept {
        uint64_t combined = (static_cast<uint64_t>(p.first) << 32) | static_cast<uint64_t>(p.second);
        combined ^= combined >> 30;
        combined *= 0xbf58476d1ce4e5b9ULL;
        combined ^= combined >> 27;
        combined *= 0x94d049bb133111ebULL;
        combined ^= combined >> 31;
        return static_cast<std::size_t>(combined);
    }
};

// Ingestion telemetry and benchmark statistics
struct IngestionStats {
    size_t file_count{0};
    size_t total_bytes{0};
    size_t total_tokens{0};
    double compression_ratio{0.0}; // bytes / token
    double elapsed_ms{0.0};
    double throughput_mb_s{0.0};
    double tokens_per_sec{0.0};
};

// Binary header structure for contiguous serialized token datasets
struct alignas(8) TokenFileHeader {
    uint32_t magic{0x54495441}; // 'TITA'
    uint32_t version{1};
    uint64_t token_count{0};
    uint32_t vocab_size{0};
    uint32_t reserved{0};
};

// Zero-copy memory-mapped binary token buffer optimized for Apple Silicon unified memory
struct MappedTokenBuffer {
    const TokenId* data{nullptr};
    size_t count{0};
    void* raw_mapping{nullptr};
    size_t raw_length{0};
    int fd{-1};

    MappedTokenBuffer() noexcept = default;
    ~MappedTokenBuffer();

    MappedTokenBuffer(const MappedTokenBuffer&) = delete;
    MappedTokenBuffer& operator=(const MappedTokenBuffer&) = delete;

    MappedTokenBuffer(MappedTokenBuffer&& other) noexcept;
    MappedTokenBuffer& operator=(MappedTokenBuffer&& other) noexcept;

    [[nodiscard]] std::span<const TokenId> span() const noexcept {
        return {data, count};
    }
    [[nodiscard]] const TokenId* data_ptr() const noexcept { return data; }
    [[nodiscard]] size_t size() const noexcept { return count; }
    [[nodiscard]] bool empty() const noexcept { return count == 0; }
    [[nodiscard]] bool is_valid() const noexcept { return data != nullptr && count > 0; }
    [[nodiscard]] TokenId operator[](size_t idx) const noexcept { return data[idx]; }
};

// High-performance BPE Tokenizer for Transformer coding models
class Tokenizer {
public:
    // Initializes tokenizer. If load_default_code_merges is true, pre-seeds
    // common source code tokens (indentation, operators, keywords).
    explicit Tokenizer(bool load_default_code_merges = true);

    // Initializes tokenizer from a saved vocabulary file
    explicit Tokenizer(const std::filesystem::path& vocab_path);

    ~Tokenizer() = default;

    Tokenizer(const Tokenizer&) = default;
    Tokenizer& operator=(const Tokenizer&) = default;
    Tokenizer(Tokenizer&&) noexcept = default;
    Tokenizer& operator=(Tokenizer&&) noexcept = default;

    // --- Core Tokenization API ---

    // Encode text/source code into contiguous flat integer tokens
    [[nodiscard]] std::vector<TokenId> encode(
        std::string_view text,
        bool add_bos = true,
        bool add_eos = true
    ) const;

    // Encode directly into an existing contiguous vector (avoids heap reallocations)
    void encode_into(
        std::string_view text,
        std::vector<TokenId>& out,
        bool add_bos = true,
        bool add_eos = true
    ) const;

    // Decode flat contiguous integer tokens back into UTF-8 source text
    [[nodiscard]] std::string decode(
        std::span<const TokenId> tokens,
        bool skip_special = true
    ) const;

    [[nodiscard]] std::string decode(
        const std::vector<TokenId>& tokens,
        bool skip_special = true
    ) const {
        return decode(std::span<const TokenId>(tokens), skip_special);
    }

    // Retrieve the exact string piece for a token ID
    [[nodiscard]] std::string token_to_piece(TokenId id) const;

    // Convert string to token ID if present in vocabulary
    [[nodiscard]] std::optional<TokenId> piece_to_token(std::string_view piece) const;

    // --- Bare-Metal Memory-Mapped File Ingestion ---

    // Ingest a single source code file using zero-copy POSIX mmap
    [[nodiscard]] std::vector<TokenId> ingest_file(
        const std::filesystem::path& filepath,
        bool add_bos = true,
        bool add_eos = true
    ) const;

    // --- Infinite Flat-Memory Streaming Tokenization Engine ---
    // Recursively steps through directories file-by-file, encodes each file
    // immediately with the active 4,096 BPE vocabulary, and streams raw
    // uint32_t token bytes directly to disk in append mode.
    // Flat memory footprint O(1) buffer — runs in constant RAM (<30MB)
    // even for 10GB or 100GB+ production datasets.
    IngestionStats stream_tokenize_directories(
        const std::vector<std::filesystem::path>& roots,
        const std::filesystem::path& output_bin,
        const std::vector<std::string>& extensions = {
            ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx", ".qml"
        },
        bool add_bos = true,
        bool add_eos = true
    ) const;

    // Phase 9: Conversational Instruction Streaming Ingestion Engine
    // Streams JSON lines of {instruction, output} conversational pairs, wraps them
    // inside the strict target chat template:
    //   "<|user|>\n{instruction}\n<|assistant|>\n{output}<|end_of_text|>"
    // and streams contiguous raw uint32_t BPE token bytes directly to disk with O(1) RAM.
    IngestionStats stream_tokenize_chat_file(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_bin
    ) const;

    // Recursively ingest a directory of source files into a contiguous flat token array
    [[nodiscard]] std::vector<TokenId> ingest_directory(
        const std::filesystem::path& dir_path,
        const std::vector<std::string>& extensions = {
            ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx",
            ".py", ".rs", ".js", ".ts", ".go", ".java",
            ".m", ".metal", ".sh", ".json", ".yaml", ".md", ".txt"
        },
        bool add_bos = true,
        bool add_eos = true
    ) const;

    // Benchmark ingestion throughput on a file or directory
    [[nodiscard]] IngestionStats benchmark_ingestion(const std::filesystem::path& path) const;

    // --- Binary Token File Serialization (for Zero-Copy DataLoader) ---

    // Save contiguous flat tokens to disk in raw binary format with header
    bool save_tokens_binary(
        const std::filesystem::path& filepath,
        std::span<const TokenId> tokens
    ) const;

    // Load contiguous flat tokens from binary file into a newly allocated vector
    [[nodiscard]] std::vector<TokenId> load_tokens_binary(
        const std::filesystem::path& filepath
    ) const;

    // Zero-copy memory-map a binary token file directly into virtual address space
    [[nodiscard]] MappedTokenBuffer mmap_tokens_binary(
        const std::filesystem::path& filepath
    ) const;

    // --- Vocabulary Training & Management ---

    // Train BPE merges directly on an in-memory source text corpus
    void train(
        std::string_view corpus,
        size_t target_vocab_size,
        size_t min_frequency = 2
    );

    // Train BPE merges across a collection of source files
    void train_from_files(
        const std::vector<std::filesystem::path>& files,
        size_t target_vocab_size,
        size_t min_frequency = 2
    );

    // Save vocabulary and merge rules to file
    bool save_vocab(const std::filesystem::path& filepath) const;

    // Load vocabulary and merge rules from file
    bool load_vocab(const std::filesystem::path& filepath);

    // Manually register a string as a single token (generates required intermediate merges)
    TokenId register_token_string(const std::string& str);

    // Vocabulary properties
    [[nodiscard]] size_t vocab_size() const noexcept { return id_to_piece_.size(); }
    [[nodiscard]] bool is_special(TokenId id) const noexcept { return id < SPECIAL_TOKENS_COUNT; }

private:
    void init_base_vocab();
    void init_default_code_merges();
    void add_merge(TokenId t1, TokenId t2, TokenId merged_id, uint32_t rank);
    [[nodiscard]] std::optional<TokenId> find_merge(TokenId t1, TokenId t2) const;
    [[nodiscard]] std::optional<uint32_t> find_merge_rank(TokenId t1, TokenId t2) const;

    // Mapping: (token_a, token_b) -> merged_token_id
    std::unordered_map<std::pair<TokenId, TokenId>, TokenId, PairHash> pair_to_id_;

    // Mapping: (token_a, token_b) -> merge_rank (lower is higher priority)
    std::unordered_map<std::pair<TokenId, TokenId>, uint32_t, PairHash> pair_to_rank_;

    // Mapping: token_id -> piece string
    std::vector<std::string> id_to_piece_;

    // Mapping: piece string -> token_id
    std::unordered_map<std::string, TokenId> piece_to_id_;

    // Monotonically increasing counter for merge priority ranks
    uint32_t next_rank_{0};
};

} // namespace titan

#endif // TITAN_TOKENIZER_HPP

