#!/bin/bash
# Usage: ./run_all_cpu.sh [threads]
# threads: Number of threads for individual benchmark (default: 1)

THREADS=${1:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

datasets=("sintel" "kitti")
oppoints=(1 2 3 4)

echo "Starting All CPU Benchmarks..."

for op in "${oppoints[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "--------------------------------------------------"
        echo "Running CPU benchmarks for Dataset: $dataset, Op: $op"
        echo "--------------------------------------------------"
        
        # Individual
        echo "Running Individual Benchmark..."
        "$SCRIPT_DIR/run_cpu_individual.sh" "$dataset" "$op" "$THREADS"
        
        # Throughput
        echo "Running Throughput Benchmark..."
        "$SCRIPT_DIR/run_cpu_throughput.sh" "$dataset" "$op"
        
        echo "Finished $dataset Op $op"
    done
done

echo "All CPU benchmarks completed."
