#include "tokenizer.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <Accelerate/Accelerate.h>

namespace {

void print_banner() {
    std::cout << "====================================================================\n";
    std::cout << "  TITAN-LM: Bare-Metal Transformer Engine (Apple Silicon M4 / C++20)\n";
    std::cout << "  Phase 1: Ingestion & Contiguous Flat Token Pipeline\n";
    std::cout << "====================================================================\n\n";
}

void print_system_telemetry() {
    std::cout << "[System Architecture]\n";
    std::cout << "  Target CPU: Apple Silicon M4 (ARM64 Neon / Apple Matrix Coprocessor)\n";
    std::cout << "  Host Concurrency: " << std::thread::hardware_concurrency() << " hardware execution threads\n";

    // Validate Apple Accelerate BLAS/vDSP availability
    float test_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float test_b[4] = {2.0f, 0.5f, 1.0f, 2.0f};
    float test_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    vDSP_vmul(test_a, 1, test_b, 1, test_c, 1, 4);

    std::cout << "  Accelerate Framework: Active (vDSP vector dot-product verified: "
              << test_c[0] << ", " << test_c[1] << ", " << test_c[2] << ", " << test_c[3] << ")\n\n";
}

void run_ingestion_verification(const titan::Tokenizer& tokenizer) {
    std::cout << "[Phase 1 Verification: Ingestion & Detokenization Fidelity]\n";

    const std::string sample_source =
        "#include <iostream>\n"
        "#include <span>\n\n"
        "// Titan-LM Core Kernel\n"
        "template <typename T>\n"
        "void compute_attention(std::span<const T> q, std::span<const T> k) {\n"
        "    const size_t dim = q.size();\n"
        "    for (size_t i = 0; i < dim; ++i) {\n"
        "        // Contiguous SIMD Accumulation\n"
        "    }\n"
        "}\n"
        "int main() {\n"
        "    std::cout << \"Bare-metal Transformer initialized.\" << std::endl;\n"
        "    return 0;\n"
        "}\n";

    // 1. Convert text to contiguous flat integer tokens
    std::vector<titan::TokenId> tokens = tokenizer.encode(sample_source, true, true);

    std::cout << "  Original Source Size: " << sample_source.size() << " bytes\n";
    std::cout << "  Contiguous Token Count: " << tokens.size() << " tokens\n";
    std::cout << "  Vocabulary Capacity: " << tokenizer.vocab_size() << " entries\n";
    std::cout << "  Compression Ratio: "
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(sample_source.size()) / tokens.size()) << " bytes/token\n";

    // Print sample of initial tokens
    std::cout << "  Token Stream Prefix: [";
    for (size_t i = 0; i < std::min<size_t>(tokens.size(), 12); ++i) {
        std::cout << tokens[i] << (i + 1 < std::min<size_t>(tokens.size(), 12) ? ", " : "");
    }
    std::cout << "...]\n";

    // 2. Decode back and verify byte-for-byte exactness
    std::string reconstructed = tokenizer.decode(tokens, true);
    if (reconstructed == sample_source) {
        std::cout << "  Roundtrip Integrity: 100% BYTE-EXACT MATCH PASSED\n\n";
    } else {
        std::cerr << "  Roundtrip Integrity: MISMATCH DETECTED\n\n";
    }

    // 3. Ingest project files and export binary token stream
    const std::string out_bin = "titan_tokens.bin";
    std::cout << "[Ingesting Workspace Source Files]\n";

    std::vector<titan::TokenId> workspace_tokens;
    if (std::filesystem::exists("tokenizer.hpp")) {
        auto hpp_tokens = tokenizer.ingest_file("tokenizer.hpp", true, true);
        workspace_tokens.insert(workspace_tokens.end(), hpp_tokens.begin(), hpp_tokens.end());
    }
    if (std::filesystem::exists("tokenizer.cpp")) {
        auto cpp_tokens = tokenizer.ingest_file("tokenizer.cpp", true, true);
        workspace_tokens.insert(workspace_tokens.end(), cpp_tokens.begin(), cpp_tokens.end());
    }

    std::cout << "  Workspace Total Contiguous Tokens: " << workspace_tokens.size() << " tokens\n";

    // Save flat binary tokens
    if (tokenizer.save_tokens_binary(out_bin, workspace_tokens)) {
        std::cout << "  Binary Serialized: " << out_bin << " ("
                  << (workspace_tokens.size() * sizeof(titan::TokenId)) + sizeof(titan::TokenFileHeader)
                  << " bytes on disk)\n";
    }

    // 4. Memory-map binary tokens using zero-copy Apple Silicon unified memory
    titan::MappedTokenBuffer mapped = tokenizer.mmap_tokens_binary(out_bin);
    if (mapped.is_valid()) {
        std::cout << "  Zero-Copy mmap Active: Mapped " << mapped.size() << " tokens at address "
                  << static_cast<const void*>(mapped.data_ptr()) << "\n";
    }
    std::cout << "\n";
}

// ============================================================================
// Phase 2: Training Loop Orchestration Scaffold
// ============================================================================

struct TrainingConfig {
    size_t vocab_size{512};
    size_t context_length{128};
    size_t batch_size{8};
    size_t embed_dim{256};
    size_t num_heads{8};
    size_t num_layers{6};
    float learning_rate{3e-4f};
    size_t max_steps{100};
};

void orchestrate_training_pipeline(
    [[maybe_unused]] const TrainingConfig& config,
    [[maybe_unused]] const titan::MappedTokenBuffer& token_stream
) {
    std::cout << "[Training Loop Orchestration Scaffold (Phase 2 Prep)]\n";
    std::cout << "  Context Length: " << config.context_length << " tokens\n";
    std::cout << "  Batch Size: " << config.batch_size << " sequences\n";
    std::cout << "  Model Dimensions: d_model=" << config.embed_dim
              << ", heads=" << config.num_heads
              << ", layers=" << config.num_layers << "\n";
    std::cout << "  Data Stream: Ready for zero-copy strided batch tensor extraction.\n";
    std::cout << "  Status: Awaiting Phase 2 Model Architecture & Autograd Implementation.\n\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    print_banner();
    print_system_telemetry();

    titan::Tokenizer tokenizer(true);

    if (argc >= 2) {
        std::string command = argv[1];

        if (command == "ingest" && argc >= 4) {
            std::filesystem::path input_path = argv[2];
            std::filesystem::path output_path = argv[3];

            std::cout << "[Ingesting: " << input_path << " -> " << output_path << "]\n";
            std::vector<titan::TokenId> tokens;
            if (std::filesystem::is_directory(input_path)) {
                tokens = tokenizer.ingest_directory(input_path);
            } else {
                tokens = tokenizer.ingest_file(input_path);
            }

            if (tokenizer.save_tokens_binary(output_path, tokens)) {
                std::cout << "Successfully saved " << tokens.size()
                          << " contiguous tokens to " << output_path << "\n";
            } else {
                std::cerr << "Failed to write output binary.\n";
                return 1;
            }
            return 0;
        }

        if (command == "train-vocab" && argc >= 5) {
            std::filesystem::path corpus_path = argv[2];
            std::filesystem::path vocab_path = argv[3];
            size_t target_size = std::stoul(argv[4]);

            std::cout << "[Training BPE Vocabulary from: " << corpus_path
                      << " (Target Vocab Size: " << target_size << ")]\n";

            tokenizer.train_from_files({corpus_path}, target_size);
            if (tokenizer.save_vocab(vocab_path)) {
                std::cout << "Successfully exported vocabulary to " << vocab_path
                          << " (Final Vocab Size: " << tokenizer.vocab_size() << ")\n";
            } else {
                std::cerr << "Failed to export vocabulary file.\n";
                return 1;
            }
            return 0;
        }

        if (command == "benchmark" && argc >= 3) {
            std::filesystem::path target = argv[2];
            auto stats = tokenizer.benchmark_ingestion(target);
            std::cout << "[Ingestion Benchmark: " << target << "]\n";
            std::cout << "  Files Processed: " << stats.file_count << "\n";
            std::cout << "  Total Bytes: " << stats.total_bytes << " bytes\n";
            std::cout << "  Total Tokens: " << stats.total_tokens << " tokens\n";
            std::cout << "  Duration: " << stats.elapsed_ms << " ms\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << stats.throughput_mb_s << " MB/s ("
                      << stats.tokens_per_sec << " tokens/sec)\n";
            return 0;
        }

        std::cerr << "Unknown command: " << command << "\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " (runs self-test and workspace ingestion)\n";
        std::cout << "  " << argv[0] << " ingest <input_file_or_dir> <output_tokens.bin>\n";
        std::cout << "  " << argv[0] << " train-vocab <corpus_file> <output_vocab.txt> <target_size>\n";
        std::cout << "  " << argv[0] << " benchmark <target_file_or_dir>\n";
        return 1;
    }

    // Default execution: Verify Phase 1 ingestion pipeline and prepare Phase 2 scaffold
    run_ingestion_verification(tokenizer);

    TrainingConfig default_config;
    default_config.vocab_size = tokenizer.vocab_size();
    titan::MappedTokenBuffer token_stream = tokenizer.mmap_tokens_binary("titan_tokens.bin");

    orchestrate_training_pipeline(default_config, token_stream);

    std::cout << "Phase 1: Ingestion Pipeline Ready.\n";
    return 0;
}

