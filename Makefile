# ==============================================================================
# Titan-LM: Bare-Metal Transformer Engine
# Hardware Target: Apple Silicon M4
# Language Standard: C++20
# Compiler: Clang (Apple LLVM)
# ==============================================================================

CXX         := clang++
CXXFLAGS    := -std=c++20 -O3 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -framework Accelerate -Wall -Wextra -Wpedantic -Wshadow -Qunused-arguments
LDFLAGS     := -framework Accelerate

TARGET      := titan_lm
SRCS        := main.cpp tokenizer.cpp tensor.cpp attention.cpp model.cpp trainer.cpp
OBJS        := $(SRCS:.cpp=.o)
DEPS        := tokenizer.hpp tensor.hpp attention.hpp model.hpp trainer.hpp

.PHONY: all clean run benchmark test help train generate

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
	@echo "==> Launching Full Titan-LM Training Loop..."
	./$(TARGET) train

generate: $(TARGET)
	@echo "==> Running Autoregressive Code Generation Stream..."
	./$(TARGET) generate "template <typename T>"

benchmark: $(TARGET)
	@echo "==> Benchmarking Ingestion & Transformer Execution on Apple Silicon M4..."
	./$(TARGET) benchmark .

clean:
	@echo "==> Cleaning build artifacts and binaries..."
	rm -f $(OBJS) $(TARGET) *.bin *.vocab *.txt

help:
	@echo "Titan-LM Build Targets:"
	@echo "  make              - Compile optimized $(TARGET) binary"
	@echo "  make run          - Execute $(TARGET) verification suite"
	@echo "  make test         - Run Phases 1-5 verification suite"
	@echo "  make train        - Execute AdamW training loop on token dataset"
	@echo "  make generate     - Run real-time streaming autoregressive code generation"
	@echo "  make benchmark    - Benchmark system throughput"
	@echo "  make clean        - Remove compiled objects and generated binaries"
