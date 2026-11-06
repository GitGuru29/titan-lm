#include "tokenizer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <cstring>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace titan {

// ============================================================================
// MappedTokenBuffer Implementation
// ============================================================================

MappedTokenBuffer::~MappedTokenBuffer() {
    if (raw_mapping != nullptr && raw_length > 0) {
        ::munmap(raw_mapping, raw_length);
        raw_mapping = nullptr;
    }
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
    data = nullptr;
    count = 0;
    raw_length = 0;
}

MappedTokenBuffer::MappedTokenBuffer(MappedTokenBuffer&& other) noexcept
    : data(other.data),
      count(other.count),
      raw_mapping(other.raw_mapping),
      raw_length(other.raw_length),
      fd(other.fd) {
    other.data = nullptr;
    other.count = 0;
    other.raw_mapping = nullptr;
    other.raw_length = 0;
    other.fd = -1;
}

MappedTokenBuffer& MappedTokenBuffer::operator=(MappedTokenBuffer&& other) noexcept {
    if (this != &other) {
        if (raw_mapping != nullptr && raw_length > 0) {
            ::munmap(raw_mapping, raw_length);
        }
        if (fd >= 0) {
            ::close(fd);
        }

        data = other.data;
        count = other.count;
        raw_mapping = other.raw_mapping;
        raw_length = other.raw_length;
        fd = other.fd;

        other.data = nullptr;
        other.count = 0;
        other.raw_mapping = nullptr;
        other.raw_length = 0;
        other.fd = -1;
    }
    return *this;
}

// ============================================================================
// Tokenizer Implementation
// ============================================================================

Tokenizer::Tokenizer(bool load_default_code_merges) {
    init_base_vocab();
    if (std::filesystem::exists("titan_vocab.txt")) {
        load_vocab("titan_vocab.txt");
    } else if (load_default_code_merges) {
        init_default_code_merges();
    }
}

Tokenizer::Tokenizer(const std::filesystem::path& vocab_path) {
    init_base_vocab();
    if (!load_vocab(vocab_path)) {
        throw std::runtime_error("Failed to load vocabulary from: " + vocab_path.string());
    }
}

void Tokenizer::init_base_vocab() {
    pair_to_id_.clear();
    pair_to_rank_.clear();
    id_to_piece_.clear();
    piece_to_id_.clear();
    next_rank_ = 0;

    id_to_piece_.resize(BASE_VOCAB_SIZE);

    // Canonical Special Tokens
    id_to_piece_[PAD_TOKEN_ID] = std::string(PAD_TOKEN_STR);
    id_to_piece_[BOS_TOKEN_ID] = std::string(BOS_TOKEN_STR);
    id_to_piece_[EOS_TOKEN_ID] = std::string(EOS_TOKEN_STR);
    id_to_piece_[UNK_TOKEN_ID] = std::string(UNK_TOKEN_STR);

    piece_to_id_[std::string(PAD_TOKEN_STR)] = PAD_TOKEN_ID;
    piece_to_id_[std::string(BOS_TOKEN_STR)] = BOS_TOKEN_ID;
    piece_to_id_[std::string(EOS_TOKEN_STR)] = EOS_TOKEN_ID;
    piece_to_id_[std::string(UNK_TOKEN_STR)] = UNK_TOKEN_ID;

    // Full 256 byte-level fallback tokens (0x00 to 0xFF)
    for (uint32_t b = 0; b < 256; ++b) {
        TokenId id = BYTE_TOKEN_OFFSET + b;
        std::string piece(1, static_cast<char>(static_cast<unsigned char>(b)));
        id_to_piece_[id] = piece;
        piece_to_id_[piece] = id;
    }
}

void Tokenizer::add_merge(TokenId t1, TokenId t2, TokenId merged_id, uint32_t rank) {
    pair_to_id_[{t1, t2}] = merged_id;
    pair_to_rank_[{t1, t2}] = rank;

    if (merged_id >= id_to_piece_.size()) {
        id_to_piece_.resize(merged_id + 1);
    }

    std::string piece = id_to_piece_[t1] + id_to_piece_[t2];
    id_to_piece_[merged_id] = piece;
    piece_to_id_[piece] = merged_id;
}

std::optional<TokenId> Tokenizer::find_merge(TokenId t1, TokenId t2) const {
    auto it = pair_to_id_.find({t1, t2});
    if (it != pair_to_id_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<uint32_t> Tokenizer::find_merge_rank(TokenId t1, TokenId t2) const {
    auto it = pair_to_rank_.find({t1, t2});
    if (it != pair_to_rank_.end()) {
        return it->second;
    }
    return std::nullopt;
}

TokenId Tokenizer::register_token_string(const std::string& str) {
    if (str.empty()) return UNK_TOKEN_ID;

    auto it = piece_to_id_.find(str);
    if (it != piece_to_id_.end()) {
        return it->second;
    }

    if (str.size() == 1) {
        return BYTE_TOKEN_OFFSET + static_cast<unsigned char>(str[0]);
    }

    // Sequentially merge components
    auto sub_tokens = encode(str, false, false);
    if (sub_tokens.size() <= 1) {
        return sub_tokens.empty() ? UNK_TOKEN_ID : sub_tokens[0];
    }

    TokenId current = sub_tokens[0];
    for (size_t i = 1; i < sub_tokens.size(); ++i) {
        TokenId next = sub_tokens[i];
        auto existing = find_merge(current, next);
        if (existing.has_value()) {
            current = *existing;
        } else {
            TokenId new_id = static_cast<TokenId>(id_to_piece_.size());
            add_merge(current, next, new_id, next_rank_++);
            current = new_id;
        }
    }
    return current;
}

void Tokenizer::init_default_code_merges() {
    // Curated high-frequency source code tokens for systems programming and C++20
    const std::vector<std::string> code_tokens = {
        // Indentation patterns (powers of two and common tab stops)
        "  ", "    ", "      ", "        ", "            ", "                ",
        "\t", "\r\n", "\n\n", "\n  ", "\n    ", "\n        ", "\n            ",

        // Operators & punctuation
        "::", "->", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=",
        "++", "--", "&&", "||", "<<", ">>", "...", "<=>", "->*", ".*",
        "//", "/*", "*/", "///", "/**", "// ", "/* ", " */",

        // Preprocessor directives
        "#include", "#include ", "#define", "#define ", "#pragma", "#pragma ",
        "#ifdef", "#ifdef ", "#ifndef", "#ifndef ", "#endif", "#if", "#else", "#elif",

        // Standard library symbols
        "std::", "std::vector", "std::vector<", "std::string", "std::string_view",
        "std::span", "std::span<", "std::cout", "std::cin", "std::endl",
        "std::move", "std::forward", "std::make_unique", "std::make_shared",
        "std::pair", "std::tuple", "std::optional", "std::unique_ptr", "std::shared_ptr",

        // Fundamental types & systems primitives
        "uint8_t", "uint8_t ", "uint16_t", "uint16_t ", "uint32_t", "uint32_t ",
        "uint64_t", "uint64_t ", "int8_t", "int16_t", "int32_t", "int32_t ",
        "int64_t", "int64_t ", "size_t", "size_t ", "float", "float ",
        "double", "double ", "bool", "bool ", "char", "char ", "void", "void ",
        "auto", "auto ", "const", "const ",

        // Keywords & semantics
        "return", "return ", "return 0;", "return true;", "return false;", "return nullptr;",
        "class", "class ", "struct", "struct ", "namespace", "namespace ",
        "template", "template<", "template <", "typename", "typename ",
        "public:", "private:", "protected:", "virtual", "virtual ", "override",
        "constexpr", "constexpr ", "consteval", "constinit", "noexcept", "noexcept ",
        "inline", "inline ", "static", "static ", "explicit", "explicit ",
        "if (", "if (", "else ", "else if ", "for (", "while (", "do {",
        "switch (", "case ", "default:", "break;", "continue;",
        "nullptr", "true", "false", "this->", "sizeof(", "alignof(",

        // Multi-language coding commonalities (Python/Rust/JS/Metal)
        "def ", "fn ", "let ", "mut ", "import ", "from ", "self.", "impl ",
        "pub ", "async ", "await ", "kernel ", "device ", "constant ", "threadgroup "
    };

    for (const auto& tok : code_tokens) {
        register_token_string(tok);
    }
}

// ============================================================================
// Encoding & Decoding Implementation
// ============================================================================

namespace {

struct BPENode {
    TokenId id{0};
    int32_t prev{-1};
    int32_t next{-1};
    bool deleted{false};
};

struct MergeCandidate {
    uint32_t rank;
    int32_t pos;

    bool operator>(const MergeCandidate& other) const noexcept {
        if (rank != other.rank) return rank > other.rank;
        return pos > other.pos;
    }
};

} // anonymous namespace

void Tokenizer::encode_into(
    std::string_view text,
    std::vector<TokenId>& out,
    bool add_bos,
    bool add_eos
) const {
    if (text.empty()) {
        if (add_bos) out.push_back(BOS_TOKEN_ID);
        if (add_eos) out.push_back(EOS_TOKEN_ID);
        return;
    }

    if (add_bos) {
        out.push_back(BOS_TOKEN_ID);
    }

    const size_t n = text.size();
    if (n == 1) {
        out.push_back(BYTE_TOKEN_OFFSET + static_cast<unsigned char>(text[0]));
        if (add_eos) out.push_back(EOS_TOKEN_ID);
        return;
    }

    // Build intrusive doubly-linked list of byte tokens
    std::vector<BPENode> nodes(n);
    for (size_t i = 0; i < n; ++i) {
        nodes[i].id = BYTE_TOKEN_OFFSET + static_cast<unsigned char>(text[i]);
        nodes[i].prev = (i > 0) ? static_cast<int32_t>(i - 1) : -1;
        nodes[i].next = (i + 1 < n) ? static_cast<int32_t>(i + 1) : -1;
        nodes[i].deleted = false;
    }

    // Min-heap tracking valid candidate merges ordered by priority rank
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<MergeCandidate>> pq;

    for (size_t i = 0; i + 1 < n; ++i) {
        auto rank_opt = find_merge_rank(nodes[i].id, nodes[i + 1].id);
        if (rank_opt.has_value()) {
            pq.push({*rank_opt, static_cast<int32_t>(i)});
        }
    }

    // Iteratively execute highest-priority BPE merges
    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();

        int32_t pos = top.pos;
        if (nodes[pos].deleted) continue;
        int32_t next_pos = nodes[pos].next;
        if (next_pos < 0 || nodes[next_pos].deleted) continue;

        auto rank_opt = find_merge_rank(nodes[pos].id, nodes[next_pos].id);
        if (!rank_opt.has_value() || *rank_opt != top.rank) continue;

        auto merged_opt = find_merge(nodes[pos].id, nodes[next_pos].id);
        if (!merged_opt.has_value()) continue;

        // Perform merge at pos
        nodes[pos].id = *merged_opt;
        nodes[next_pos].deleted = true;

        int32_t next_next = nodes[next_pos].next;
        nodes[pos].next = next_next;
        if (next_next >= 0) {
            nodes[next_next].prev = pos;
        }

        // Check newly formed adjacent pairs
        if (nodes[pos].prev >= 0) {
            auto left_rank = find_merge_rank(nodes[nodes[pos].prev].id, nodes[pos].id);
            if (left_rank.has_value()) {
                pq.push({*left_rank, nodes[pos].prev});
            }
        }
        if (nodes[pos].next >= 0) {
            auto right_rank = find_merge_rank(nodes[pos].id, nodes[nodes[pos].next].id);
            if (right_rank.has_value()) {
                pq.push({*right_rank, pos});
            }
        }
    }

    // Traverse linked list and emit surviving tokens
    int32_t curr = 0;
    while (curr >= 0 && nodes[curr].deleted) {
        curr = nodes[curr].next;
    }

    while (curr >= 0) {
        if (!nodes[curr].deleted) {
            out.push_back(nodes[curr].id);
        }
        curr = nodes[curr].next;
    }

    if (add_eos) {
        out.push_back(EOS_TOKEN_ID);
    }
}

std::vector<TokenId> Tokenizer::encode(
    std::string_view text,
    bool add_bos,
    bool add_eos
) const {
    std::vector<TokenId> tokens;
    // Pre-reserve estimated compressed token capacity
    tokens.reserve(text.size() / 2 + 4);
    encode_into(text, tokens, add_bos, add_eos);
    return tokens;
}

std::string Tokenizer::decode(
    std::span<const TokenId> tokens,
    bool skip_special
) const {
    std::string result;
    result.reserve(tokens.size() * 3); // Average byte length per token estimation

    for (TokenId id : tokens) {
        if (skip_special && is_special(id)) {
            continue;
        }
        if (id < id_to_piece_.size()) {
            result.append(id_to_piece_[id]);
        } else {
            result.append(UNK_TOKEN_STR);
        }
    }
    return result;
}

std::string Tokenizer::token_to_piece(TokenId id) const {
    if (id < id_to_piece_.size()) {
        return id_to_piece_[id];
    }
    return std::string(UNK_TOKEN_STR);
}

std::optional<TokenId> Tokenizer::piece_to_token(std::string_view piece) const {
    auto it = piece_to_id_.find(std::string(piece));
    if (it != piece_to_id_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ============================================================================
// Bare-Metal Memory-Mapped File Ingestion
// ============================================================================

std::vector<TokenId> Tokenizer::ingest_file(
    const std::filesystem::path& filepath,
    bool add_bos,
    bool add_eos
) const {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("File not found: " + filepath.string());
    }

    const auto file_size = std::filesystem::file_size(filepath);
    if (file_size == 0) {
        std::vector<TokenId> empty_tokens;
        if (add_bos) empty_tokens.push_back(BOS_TOKEN_ID);
        if (add_eos) empty_tokens.push_back(EOS_TOKEN_ID);
        return empty_tokens;
    }

    int fd = ::open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open file: " + filepath.string() + " (" + std::strerror(errno) + ")");
    }

    // Zero-copy POSIX memory mapping
    void* addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("Failed to mmap file: " + filepath.string() + " (" + std::strerror(errno) + ")");
    }

    // Optimize sequential page prefetching on Apple Silicon unified memory
    ::madvise(addr, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);

    std::string_view view(static_cast<const char*>(addr), file_size);
    auto tokens = encode(view, add_bos, add_eos);

    ::munmap(addr, file_size);
    ::close(fd);

    return tokens;
}

std::vector<TokenId> Tokenizer::ingest_directory(
    const std::filesystem::path& dir_path,
    const std::vector<std::string>& extensions,
    bool add_bos,
    bool add_eos
) const {
    if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
        throw std::runtime_error("Directory not found: " + dir_path.string());
    }

    // Collect all matching file paths deterministically
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            for (const auto& target_ext : extensions) {
                if (ext == target_ext) {
                    files.push_back(entry.path());
                    break;
                }
            }
        }
    }

    std::sort(files.begin(), files.end());

    std::vector<TokenId> flat_corpus_tokens;
    for (const auto& file : files) {
        try {
            encode_into(
                std::string_view(), // handled inside ingest_file
                flat_corpus_tokens,
                false, false
            );
            auto file_tokens = ingest_file(file, add_bos, add_eos);
            flat_corpus_tokens.insert(flat_corpus_tokens.end(), file_tokens.begin(), file_tokens.end());
        } catch (const std::exception& e) {
            std::cerr << "[Titan-Tokenizer] Warning: Failed to ingest " << file << ": " << e.what() << "\n";
        }
    }

    return flat_corpus_tokens;
}

IngestionStats Tokenizer::stream_tokenize_directories(
    const std::vector<std::filesystem::path>& roots,
    const std::filesystem::path& output_bin,
    const std::vector<std::string>& extensions,
    bool add_bos,
    bool add_eos
) const {
    IngestionStats stats{};
    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream out(output_bin, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output binary for streaming: " + output_bin.string());
    }

    // 1. Write placeholder header (will be overwritten at byte 0 upon completion)
    TokenFileHeader placeholder_header{
        .magic = 0x54495441, // 'TITA'
        .version = 1,
        .token_count = 0,
        .vocab_size = static_cast<uint32_t>(vocab_size()),
        .reserved = 0
    };
    out.write(reinterpret_cast<const char*>(&placeholder_header), sizeof(placeholder_header));

    // Flat memory strategy: reusable heap buffers to guarantee O(1) RAM
    std::string file_buffer;
    std::vector<TokenId> token_buffer;
    token_buffer.reserve(32768);

    uint64_t total_tokens = 0;
    uint64_t total_bytes = 0;
    size_t file_count = 0;

    std::unordered_set<std::string> ext_set(extensions.begin(), extensions.end());

    std::cout << "====================================================================\n";
    std::cout << "  TITAN-LM Stream Ingestion Engine: Infinite Memory-Flat Pipeline\n";
    std::cout << "====================================================================\n";
    std::cout << "  Target Binary : " << output_bin << "\n";
    std::cout << "  Active Vocab  : " << vocab_size() << " entries (BPE mapped)\n";
    std::cout << "  RAM Strategy  : Constant flat footprint O(1) — zero text file merge\n";
    std::cout << "--------------------------------------------------------------------\n";

    for (const auto& root : roots) {
        if (!std::filesystem::exists(root)) {
            std::cerr << "[Stream Ingest] Warning: Directory does not exist: " << root << " (skipping)\n";
            continue;
        }

        std::cout << "[Stream Ingest] Scanning codebase: " << root << " ...\n";

        std::error_code ec;
        for (std::filesystem::recursive_directory_iterator it(root, std::filesystem::directory_options::skip_permission_denied, ec), end;
             it != end; it.increment(ec)) {
            if (ec) {
                ec.clear();
                continue;
            }

            const auto& entry = *it;

            // Fast path: skip massive build & git internal directories
            if (entry.is_directory()) {
                auto name = entry.path().filename().string();
                if (name == ".git" || name == "build" || name == "CMakeFiles" ||
                    name == ".gemini" || name == "node_modules" || name == ".idea" ||
                    name == ".vscode" || name == "testdata") {
                    it.disable_recursion_pending();
                    continue;
                }
            }

            if (!entry.is_regular_file(ec) || ec) {
                ec.clear();
                continue;
            }

            std::string ext = entry.path().extension().string();
            if (ext_set.find(ext) == ext_set.end()) {
                continue;
            }

            uintmax_t fsize = entry.file_size(ec);
            if (ec || fsize == 0 || fsize > 10 * 1024 * 1024) { // Skip 0B or >10MB binary blobs
                ec.clear();
                continue;
            }

            // Read file into flat buffer
            std::ifstream fin(entry.path(), std::ios::binary);
            if (!fin) continue;

            file_buffer.assign((std::istreambuf_iterator<char>(fin)),
                                std::istreambuf_iterator<char>());
            fin.close();

            if (file_buffer.empty()) continue;

            total_bytes += file_buffer.size();

            // Encode immediately into contiguous token buffer
            token_buffer.clear();
            encode_into(file_buffer, token_buffer, add_bos, add_eos);

            // Stream raw uint32_t tokens directly to disk in append mode
            if (!token_buffer.empty()) {
                out.write(reinterpret_cast<const char*>(token_buffer.data()),
                          token_buffer.size() * sizeof(TokenId));
                total_tokens += token_buffer.size();
            }

            file_count++;

            // Live progress report every 2,500 files
            if (file_count % 2500 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double el = std::chrono::duration<double>(now - start_time).count();
                double mb = total_bytes / (1024.0 * 1024.0);
                double tok_per_s = (el > 0) ? (total_tokens / el) : 0;
                double mb_per_s = (el > 0) ? (mb / el) : 0;

                std::cout << "  [Stream Engine] " << std::setw(6) << file_count << " files | "
                          << std::fixed << std::setprecision(1) << mb << " MB read | "
                          << (total_tokens / 1000000.0) << "M tokens streamed ("
                          << std::setprecision(1) << tok_per_s / 1000.0 << "K tok/s, "
                          << mb_per_s << " MB/s)\n";
            }
        }
    }

    // 2. Finalize header at byte 0 with exact cumulative token count
    out.seekp(0, std::ios::beg);
    TokenFileHeader final_header{
        .magic = 0x54495441, // 'TITA'
        .version = 1,
        .token_count = total_tokens,
        .vocab_size = static_cast<uint32_t>(vocab_size()),
        .reserved = 0
    };
    out.write(reinterpret_cast<const char*>(&final_header), sizeof(final_header));
    out.flush();
    out.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    stats.file_count = file_count;
    stats.total_bytes = total_bytes;
    stats.total_tokens = total_tokens;
    stats.elapsed_ms = elapsed_s * 1000.0;
    stats.compression_ratio = (total_tokens > 0) ? (static_cast<double>(total_bytes) / total_tokens) : 0.0;
    stats.throughput_mb_s = (elapsed_s > 0) ? ((total_bytes / (1024.0 * 1024.0)) / elapsed_s) : 0.0;
    stats.tokens_per_sec = (elapsed_s > 0) ? (total_tokens / elapsed_s) : 0.0;

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "[Stream Engine Completed]\n";
    std::cout << "  Total Files Processed : " << stats.file_count << "\n";
    std::cout << "  Total Raw Code Read   : " << std::fixed << std::setprecision(2)
              << (stats.total_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Total Tokens Streamed : " << stats.total_tokens << " tokens ("
              << (stats.total_tokens * sizeof(TokenId) / 1024.0 / 1024.0) << " MB binary)\n";
    std::cout << "  Token Density Ratio   : " << std::fixed << std::setprecision(2)
              << stats.compression_ratio << " bytes/token\n";
    std::cout << "  Ingestion Throughput  : " << std::fixed << std::setprecision(2)
              << stats.throughput_mb_s << " MB/s (" << (stats.tokens_per_sec / 1000.0) << "K tok/s)\n";
    std::cout << "  Elapsed Wall Time     : " << std::fixed << std::setprecision(2) << elapsed_s << " seconds\n";
    std::cout << "====================================================================\n\n";

    return stats;
}

namespace {
std::string extract_json_field_value(std::string_view json, std::string_view key) {
    std::string needle = "\"" + std::string(key) + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string_view::npos) return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string_view::npos) return "";
    size_t qstart = json.find('\"', colon + 1);
    if (qstart == std::string_view::npos) return "";

    std::string result;
    result.reserve(256);
    bool escaped = false;
    for (size_t i = qstart + 1; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            switch (c) {
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case 'b': result += '\b'; break;
                case 'f': result += '\f'; break;
                case '\"': result += '\"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'u': {
                    if (i + 4 < json.size()) {
                        i += 4;
                    }
                    break;
                }
                default: result += c; break;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '\"') {
            break;
        } else {
            result += c;
        }
    }
    return result;
}
} // anonymous namespace

IngestionStats Tokenizer::stream_tokenize_chat_file(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_bin,
    size_t seq_len
) const {
    IngestionStats stats{};
    auto start_time = std::chrono::high_resolution_clock::now();

    std::ifstream in(input_path);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Cannot open instruction file: " << input_path << "\n";
        return stats;
    }

    std::ofstream out(output_bin, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[ERROR] Cannot open output token file: " << output_bin << "\n";
        return stats;
    }

    // 1. Write placeholder header
    TokenFileHeader placeholder_header{
        .magic = 0x54495441, // 'TITA'
        .version = 1,
        .token_count = 0,
        .vocab_size = static_cast<uint32_t>(vocab_size()),
        .reserved = 0
    };
    out.write(reinterpret_cast<const char*>(&placeholder_header), sizeof(placeholder_header));

    std::cout << "\n====================================================================\n";
    std::cout << "  TITAN-LM SFT Paired Streaming Tokenizer (Turn-Aligned & Masked)\n";
    std::cout << "====================================================================\n";
    std::cout << "  Input File    : " << input_path << "\n";
    std::cout << "  Target Binary : " << output_bin << "\n";
    std::cout << "  Record Format : Turn-Aligned [X (len=" << seq_len << "), Y_target (masked prompt)]\n";
    std::cout << "  RAM Strategy  : Flat constant memory O(1) buffer\n";
    std::cout << "--------------------------------------------------------------------\n";

    std::string line;
    std::vector<uint32_t> X(seq_len, PAD_TOKEN_ID);
    std::vector<uint32_t> Y(seq_len, 0xFFFFFFFF); // 0xFFFFFFFF = IGNORE_INDEX
    size_t total_tokens = 0;
    size_t total_bytes = 0;
    size_t pair_count = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        total_bytes += line.size() + 1;

        std::string instruction;
        std::string output;

        // Extract instruction and output fields from JSON or JSONL row
        if (line.front() == '{' || line.find("\"instruction\"") != std::string::npos) {
            instruction = extract_json_field_value(line, "instruction");
            if (instruction.empty()) {
                instruction = extract_json_field_value(line, "prompt");
            }
            std::string input_ctx = extract_json_field_value(line, "input");
            if (!input_ctx.empty()) {
                instruction += "\n\nContext:\n" + input_ctx;
            }
            output = extract_json_field_value(line, "output");
            if (output.empty()) {
                output = extract_json_field_value(line, "response");
            }
        }

        if (instruction.empty() || output.empty()) continue;

        // 1. Separate Prompt and Response strings
        std::string prompt_str = "<|user|>\n" + instruction + "\n<|assistant|>\n";
        std::string resp_str = output + "<|end_of_text|>";

        std::vector<TokenId> prompt_tokens = encode(prompt_str, false, false);
        std::vector<TokenId> resp_tokens = encode(resp_str, false, false);

        if (prompt_tokens.empty() || resp_tokens.empty()) continue;

        // Combine into full sequence
        std::vector<uint32_t> full_seq = prompt_tokens;
        full_seq.insert(full_seq.end(), resp_tokens.begin(), resp_tokens.end());

        // Skip dialogues whose prompt alone exceeds sequence length
        if (prompt_tokens.size() >= seq_len) continue;

        if (full_seq.size() > seq_len) {
            full_seq.resize(seq_len);
        }

        // Initialize X with PAD, Y with IGNORE_INDEX (0xFFFFFFFF)
        std::fill(X.begin(), X.end(), PAD_TOKEN_ID);
        std::fill(Y.begin(), Y.end(), 0xFFFFFFFF);

        for (size_t i = 0; i < full_seq.size(); ++i) {
            X[i] = full_seq[i];
        }

        // Causal next-token prediction targets with prompt loss masking:
        // Position i in X predicts full_seq[i + 1].
        // Assistant response starts at index prompt_tokens.size().
        // Position (prompt_tokens.size() - 1) predicts the first assistant token!
        const size_t prompt_last_idx = (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;
        for (size_t i = 0; i + 1 < full_seq.size(); ++i) {
            if (i >= prompt_last_idx) {
                // Inside assistant response turn: COMPUTE LOSS!
                Y[i] = full_seq[i + 1];
            } else {
                // Inside user prompt turn: MASK LOSS!
                Y[i] = 0xFFFFFFFF;
            }
        }

        // Write paired record [X (seq_len), Y (seq_len)] directly to disk
        out.write(reinterpret_cast<const char*>(X.data()), seq_len * sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(Y.data()), seq_len * sizeof(uint32_t));

        total_tokens += 2 * seq_len;
        pair_count++;

        if (pair_count % 2500 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration<double>(now - start_time).count();
            double mb = total_bytes / (1024.0 * 1024.0);
            double tok_per_s = (el > 0) ? (total_tokens / el) : 0;
            double mb_per_s = (el > 0) ? (mb / el) : 0;

            std::cout << "  [Stream Instruct] " << std::setw(6) << pair_count << " pairs | "
                      << std::fixed << std::setprecision(1) << mb << " MB read | "
                      << (total_tokens / 1000.0) << "K tokens streamed ("
                      << std::setprecision(1) << tok_per_s / 1000.0 << "K tok/s, "
                      << mb_per_s << " MB/s)\n";
        }
    }

    in.close();

    // 2. Finalize header at byte 0 with exact cumulative token count
    out.seekp(0, std::ios::beg);
    TokenFileHeader final_header{
        .magic = 0x54495441,
        .version = 1,
        .token_count = total_tokens,
        .vocab_size = static_cast<uint32_t>(vocab_size()),
        .reserved = 0
    };
    out.write(reinterpret_cast<const char*>(&final_header), sizeof(final_header));
    out.flush();
    out.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    stats.file_count = pair_count;
    stats.total_bytes = total_bytes;
    stats.total_tokens = total_tokens;
    stats.elapsed_ms = elapsed_s * 1000.0;
    stats.compression_ratio = (total_tokens > 0) ? (static_cast<double>(total_bytes) / total_tokens) : 0.0;
    stats.throughput_mb_s = (elapsed_s > 0) ? ((total_bytes / (1024.0 * 1024.0)) / elapsed_s) : 0.0;
    stats.tokens_per_sec = (elapsed_s > 0) ? (total_tokens / elapsed_s) : 0.0;

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "[Stream Instruct Completed]\n";
    std::cout << "  Total Dialogue Pairs  : " << pair_count << " pairs\n";
    std::cout << "  Total Raw Text Read   : " << std::fixed << std::setprecision(2)
              << (stats.total_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Total Tokens Streamed : " << stats.total_tokens << " tokens ("
              << (stats.total_tokens * sizeof(TokenId) / 1024.0 / 1024.0) << " MB binary)\n";
    std::cout << "  Ingestion Speed       : " << std::fixed << std::setprecision(2)
              << stats.throughput_mb_s << " MB/s (" << (stats.tokens_per_sec / 1000.0) << "K tok/s)\n";
    std::cout << "  Elapsed Wall Time     : " << std::fixed << std::setprecision(2) << elapsed_s << " seconds\n";
    std::cout << "====================================================================\n\n";

    return stats;
}

IngestionStats Tokenizer::benchmark_ingestion(const std::filesystem::path& path) const {
    IngestionStats stats{};
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<TokenId> tokens;
    if (std::filesystem::is_directory(path)) {
        tokens = ingest_directory(path);
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                stats.file_count++;
                stats.total_bytes += entry.file_size();
            }
        }
    } else {
        stats.file_count = 1;
        stats.total_bytes = std::filesystem::file_size(path);
        tokens = ingest_file(path);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    stats.total_tokens = tokens.size();
    stats.elapsed_ms = duration.count();
    stats.compression_ratio = (stats.total_tokens > 0)
        ? static_cast<double>(stats.total_bytes) / static_cast<double>(stats.total_tokens)
        : 0.0;
    stats.throughput_mb_s = (stats.elapsed_ms > 0.0)
        ? (static_cast<double>(stats.total_bytes) / (1024.0 * 1024.0)) / (stats.elapsed_ms / 1000.0)
        : 0.0;
    stats.tokens_per_sec = (stats.elapsed_ms > 0.0)
        ? static_cast<double>(stats.total_tokens) / (stats.elapsed_ms / 1000.0)
        : 0.0;

    return stats;
}

// ============================================================================
// Flat Binary Token File Serialization (Zero-Copy DataLoader Target)
// ============================================================================

bool Tokenizer::save_tokens_binary(
    const std::filesystem::path& filepath,
    std::span<const TokenId> tokens
) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;

    TokenFileHeader header{
        .magic = 0x54495441, // 'TITA'
        .version = 1,
        .token_count = static_cast<uint64_t>(tokens.size()),
        .vocab_size = static_cast<uint32_t>(vocab_size()),
        .reserved = 0
    };

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(TokenId));
    return out.good();
}

std::vector<TokenId> Tokenizer::load_tokens_binary(
    const std::filesystem::path& filepath
) const {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open binary token file: " + filepath.string());
    }

    TokenFileHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good() || header.magic != 0x54495441) {
        throw std::runtime_error("Invalid or corrupted Titan token file header: " + filepath.string());
    }

    std::vector<TokenId> tokens(header.token_count);
    in.read(reinterpret_cast<char*>(tokens.data()), header.token_count * sizeof(TokenId));
    if (!in.good()) {
        throw std::runtime_error("Incomplete read of tokens from: " + filepath.string());
    }

    return tokens;
}

MappedTokenBuffer Tokenizer::mmap_tokens_binary(
    const std::filesystem::path& filepath
) const {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("Token file not found: " + filepath.string());
    }

    const auto file_size = std::filesystem::file_size(filepath);
    if (file_size < sizeof(TokenFileHeader)) {
        throw std::runtime_error("Token file too small to contain header: " + filepath.string());
    }

    int fd = ::open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open binary token file for mmap: " + filepath.string());
    }

    void* addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("Failed to mmap token file: " + filepath.string());
    }

    ::madvise(addr, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);

    const auto* header = reinterpret_cast<const TokenFileHeader*>(addr);
    if (header->magic != 0x54495441) {
        ::munmap(addr, file_size);
        ::close(fd);
        throw std::runtime_error("Corrupted magic in mmap token file: " + filepath.string());
    }

    const size_t expected_size = sizeof(TokenFileHeader) + (header->token_count * sizeof(TokenId));
    if (file_size < expected_size) {
        ::munmap(addr, file_size);
        ::close(fd);
        throw std::runtime_error("Token file truncated: expected " + std::to_string(expected_size) +
                                 " bytes, got " + std::to_string(file_size));
    }

    MappedTokenBuffer buf;
    buf.data = reinterpret_cast<const TokenId*>(static_cast<const char*>(addr) + sizeof(TokenFileHeader));
    buf.count = header->token_count;
    buf.raw_mapping = addr;
    buf.raw_length = file_size;
    buf.fd = fd;

    return buf;
}

// ============================================================================
// BPE Vocabulary Training & Management
// ============================================================================

void Tokenizer::train(
    std::string_view corpus,
    size_t target_vocab_size,
    size_t min_frequency
) {
    if (corpus.empty()) return;

    // Reset base vocabulary and preload foundational code merges
    init_base_vocab();
    init_default_code_merges();

    if (target_vocab_size <= id_to_piece_.size()) return;

    // Take a representative sample (up to 2MB) for swift, high-density BPE merge discovery
    constexpr size_t MAX_TRAIN_CORPUS_BYTES = 2 * 1024 * 1024;
    std::string_view sample = (corpus.size() > MAX_TRAIN_CORPUS_BYTES)
        ? corpus.substr(0, MAX_TRAIN_CORPUS_BYTES)
        : corpus;

    // Initial tokenization using base & default code merges
    std::vector<TokenId> tokens = encode(sample, false, false);

    std::cout << "[Tokenizer::train] Base vocab: " << id_to_piece_.size()
              << " entries | Training Sample: " << (sample.size() / 1024.0) << " KB ("
              << tokens.size() << " initial tokens)\n";
    std::cout << "[Tokenizer::train] Scaling vocabulary up to " << target_vocab_size << " entries...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::unordered_map<std::pair<TokenId, TokenId>, size_t, PairHash> pair_counts;
    pair_counts.reserve(65536);

    while (id_to_piece_.size() < target_vocab_size && tokens.size() >= 2) {
        pair_counts.clear();
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            pair_counts[{tokens[i], tokens[i + 1]}]++;
        }

        if (pair_counts.empty()) break;

        size_t best_count = 0;
        std::pair<TokenId, TokenId> best_pair{0, 0};
        for (const auto& [pair, count] : pair_counts) {
            if (count > best_count) {
                best_count = count;
                best_pair = pair;
            }
        }

        if (best_count < min_frequency) {
            if (min_frequency > 1) {
                min_frequency = 1; // Relax threshold to reach exact target capacity
                continue;
            }
            std::cout << "[Tokenizer::train] Min frequency reached at " << id_to_piece_.size() << " tokens.\n";
            break;
        }

        TokenId new_id = static_cast<TokenId>(id_to_piece_.size());
        add_merge(best_pair.first, best_pair.second, new_id, next_rank_++);

        // Linear in-place token replacement
        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < tokens.size(); ++read_idx) {
            if (read_idx + 1 < tokens.size() &&
                tokens[read_idx] == best_pair.first &&
                tokens[read_idx + 1] == best_pair.second) {
                tokens[write_idx++] = new_id;
                read_idx++; // Skip consumed adjacent token
            } else {
                tokens[write_idx++] = tokens[read_idx];
            }
        }
        tokens.resize(write_idx);

        if (id_to_piece_.size() % 500 == 0 || id_to_piece_.size() == target_vocab_size) {
            std::cout << "  [BPE Merge] " << id_to_piece_.size() << " / " << target_vocab_size
                      << " tokens (top pair freq: " << best_count << ")\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "[Tokenizer::train] Complete: " << id_to_piece_.size() << " total tokens in "
              << std::fixed << std::setprecision(2) << elapsed_s << "s.\n";

    save_vocab("titan_vocab.txt");
    std::cout << "[Tokenizer::train] Saved vocabulary to titan_vocab.txt\n";
}

void Tokenizer::train_from_files(
    const std::vector<std::filesystem::path>& files,
    size_t target_vocab_size,
    size_t min_frequency
) {
    std::string aggregate_corpus;
    for (const auto& file : files) {
        std::ifstream in(file, std::ios::binary);
        if (in) {
            std::ostringstream ss;
            ss << in.rdbuf();
            aggregate_corpus.append(ss.str());
            aggregate_corpus.push_back('\n');
            if (aggregate_corpus.size() > 3 * 1024 * 1024) break;
        }
    }
    train(aggregate_corpus, target_vocab_size, min_frequency);
}

bool Tokenizer::save_vocab(const std::filesystem::path& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "# TITAN-LM VOCABULARY V1\n";
    out << "# MERGES_COUNT " << pair_to_id_.size() << "\n";

    // Sort merges by rank for clean deterministic export
    std::vector<std::pair<std::pair<TokenId, TokenId>, uint32_t>> sorted_merges(
        pair_to_rank_.begin(), pair_to_rank_.end()
    );
    std::sort(sorted_merges.begin(), sorted_merges.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    for (const auto& [pair, rank] : sorted_merges) {
        auto it = pair_to_id_.find(pair);
        if (it != pair_to_id_.end()) {
            out << pair.first << " " << pair.second << " " << it->second << " " << rank << "\n";
        }
    }
    return out.good();
}

bool Tokenizer::load_vocab(const std::filesystem::path& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) return false;

    init_base_vocab();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        TokenId t1{0}, t2{0}, merged_id{0};
        uint32_t rank{0};
        if (ss >> t1 >> t2 >> merged_id >> rank) {
            add_merge(t1, t2, merged_id, rank);
            if (rank >= next_rank_) {
                next_rank_ = rank + 1;
            }
        }
    }
    return true;
}

} // namespace titan

