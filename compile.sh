#!/bin/bash
# Usage: ./compile.sh [cpu|cuda|all]
# Default: all

TARGET=${1:-all}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_cpu() {
    echo "Building for CPU..."
    mkdir -p "$SCRIPT_DIR/build_cpu"
    cd "$SCRIPT_DIR/build_cpu"
    cmake -DUSE_CUDA=OFF ..
    make -j$(nproc)
    cd "$SCRIPT_DIR"
}

build_cuda() {
    echo "Building for CUDA..."
    mkdir -p "$SCRIPT_DIR/build_cuda"
    cd "$SCRIPT_DIR/build_cuda"
    cmake -DUSE_CUDA=ON ..
    make -j$(nproc)
    cd "$SCRIPT_DIR"
}

if [ "$TARGET" == "cpu" ]; then
    build_cpu
elif [ "$TARGET" == "cuda" ]; then
    build_cuda
elif [ "$TARGET" == "all" ]; then
    build_cpu
    build_cuda
else
    echo "Unknown target: $TARGET"
    echo "Usage: ./compile.sh [cpu|cuda|all]"
    exit 1
fi

echo "Build complete."
