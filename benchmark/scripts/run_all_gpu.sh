#!/bin/bash
# Usage: ./run_all_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

datasets=("sintel" "kitti")
oppoints=(1 2 3 4)

# Function to kill all background processes
cleanup() {
    echo ""
    echo "Caught signal! Killing all background processes..."
    if [ -n "$pids" ]; then
        # Kill the process group of the background jobs if possible, or just the pids
        kill $pids 2>/dev/null
    fi
    exit 1
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "Starting All GPU Benchmarks..."
echo "Launching parallel jobs for each Operation Point (mapped to GPUs)..."

pids=""

for op in "${oppoints[@]}"; do
    (
        echo "[GPU Op $op] Starting..."
        for dataset in "${datasets[@]}"; do
            echo "[GPU Op $op] Running Individual Benchmark for $dataset..."
            "$SCRIPT_DIR/run_gpu_individual.sh" "$dataset" "$op" > /dev/null
            
            echo "[GPU Op $op] Running Throughput Benchmark for $dataset..."
            "$SCRIPT_DIR/run_gpu_throughput.sh" "$dataset" "$op" > /dev/null
        done
        echo "[GPU Op $op] Finished."
    ) &
    pids="$pids $!"
done

echo "All jobs launched. Waiting for completion..."
wait $pids

echo "All GPU benchmarks completed."
