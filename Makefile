# ==============================================================================
# Titan-LM: Bare-Metal Transformer Engine
# Hardware Target: Apple Silicon M4
# Language Standard: C++20
# Compiler: Clang (Apple LLVM)
# ==============================================================================

CXX         := clang++
CXXFLAGS    := -std=c++20 -O3 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -framework Accelerate -Wall -Wextra -Wpedantic -Wshadow -Qunused-arguments
LDFLAGS     := -framework Accelerate -framework IOKit -framework CoreFoundation -framework CoreGraphics

TARGET      := titan_lm
SRCS        := main.cpp tokenizer.cpp tensor.cpp attention.cpp model.cpp trainer.cpp governor.cpp
OBJS        := $(SRCS:.cpp=.o)
DEPS        := tokenizer.hpp tensor.hpp attention.hpp model.hpp trainer.hpp governor.hpp

.PHONY: all clean run benchmark test help train generate ingest

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "==> Linking $(TARGET) [Apple Silicon M4 Native + Full Titan Engine]..."
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "==> Build complete: $(TARGET)"

%.o: %.cpp $(DEPS)
	@echo "==> Compiling $< [-mcpu=apple-m4 -O3]..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	@echo "==> Executing $(TARGET)..."
	./$(TARGET)

test: $(TARGET)
	@echo "==> Running Titan-LM Complete Verification Suite (Phases 1-5)..."
	./$(TARGET)

train: $(TARGET)
	@echo "==> Launching Titan-450M Training Loop (d_model=1152, 28 layers, 16 heads, 13.8GB RAM)..."
	./$(TARGET) train

train-instruct: $(TARGET)
	@echo "==> Launching Phase 9 Instruction Fine-Tuning & Alignment Loop..."
	./$(TARGET) train_instruct

ingest: $(TARGET)
	@echo "==> Running Data Ingestion Pipeline..."
	bash ingest.sh

ingest-instruct: $(TARGET)
	@echo "==> Running Conversational Instruction Ingestion Pipeline..."
	bash ingest_instruct.sh

chat: $(TARGET)
	@echo "==> Interactive Conversational Dialogue..."
	./$(TARGET) chat "Write a clean C++ template for a thread-safe singleton."

generate: $(TARGET)
	@echo "==> Running Autoregressive Code Generation Stream..."
	./$(TARGET) generate "int main()"

benchmark: $(TARGET)
	@echo "==> Benchmarking Ingestion & Transformer Execution on Apple Silicon M4..."
	./$(TARGET) benchmark .

clean:
	@echo "==> Cleaning build artifacts and binaries..."
	rm -f $(OBJS) $(TARGET)

help:
	@echo "Titan-LM Build Targets:"
	@echo "  make              - Compile optimized $(TARGET) binary"
	@echo "  make ingest       - Harvest source files + tokenize → titan_tokens.bin"
	@echo "  make train        - Launch AdamW training loop (d_model=256, 6L, 2000 steps)"
	@echo "  make generate     - Stream autoregressive C++ code to stdout"
	@echo "  make test         - Run Phases 1-5 verification suite"
	@echo "  make benchmark    - Benchmark system throughput"
	@echo "  make clean        - Remove compiled objects and binaries"
