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
SRCS        := main.cpp tokenizer.cpp tensor.cpp attention.cpp model.cpp
OBJS        := $(SRCS:.cpp=.o)
DEPS        := tokenizer.hpp tensor.hpp attention.hpp model.hpp

.PHONY: all clean run benchmark test help

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "==> Linking $(TARGET) [Apple Silicon M4 Native + Transformer Core]..."
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "==> Build complete: $(TARGET)"

%.o: %.cpp $(DEPS)
	@echo "==> Compiling $< [-mcpu=apple-m4 -O3]..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	@echo "==> Executing $(TARGET)..."
	./$(TARGET)

test: $(TARGET)
	@echo "==> Running Titan-LM Core Verification Suite (Phase 1, 2, 3 & 4)..."
	./$(TARGET)

benchmark: $(TARGET)
	@echo "==> Benchmarking Ingestion & Transformer Blocks on Apple Silicon M4..."
	./$(TARGET) benchmark .

clean:
	@echo "==> Cleaning build artifacts and token binaries..."
	rm -f $(OBJS) $(TARGET) *.bin *.vocab *.txt

help:
	@echo "Titan-LM Build Targets:"
	@echo "  make              - Compile optimized $(TARGET) binary"
	@echo "  make run          - Execute $(TARGET) verification and pipeline demo"
	@echo "  make test         - Run Phase 1, 2, 3 & 4 verification suite"
	@echo "  make benchmark    - Benchmark ingestion throughput & Transformer blocks"
	@echo "  make clean        - Remove compiled objects and generated binaries"
