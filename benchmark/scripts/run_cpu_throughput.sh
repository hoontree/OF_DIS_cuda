#!/bin/bash
# Usage: ./run_cpu_throughput.sh <dataset> <oppoint>

source "$(dirname "$0")/common.sh"

DATASET=$1
OPPOINT=${2:-2}

BUILD_DIR="$BENCHMARK_DIR/../build_cpu"
EXEC_NAME="run_OF_INT_PARALLEL"
OUT_DIR="$BENCHMARK_DIR/results/cpu_throughput_${DATASET}_op${OPPOINT}"
LIST_FILE="temp_pair_list_cpu_batch.txt"

mkdir -p "$OUT_DIR"
rm -f "$LIST_FILE"

# Collect pairs
echo "Collecting pairs for $DATASET..."
if [ "$DATASET" == "sintel" ]; then
    collect_pairs_sintel "$LIST_FILE"
elif [ "$DATASET" == "kitti" ]; then
    collect_pairs_kitti "$LIST_FILE"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

total_pairs=$(wc -l < "$LIST_FILE")
echo "Running CPU throughput benchmark on $total_pairs pairs from $DATASET..."

start_time=$(date +%s%N)
"$BUILD_DIR/$EXEC_NAME" "$LIST_FILE" "$OUT_DIR" "$OPPOINT" > /dev/null
end_time=$(date +%s%N)

duration_ns=$((end_time - start_time))
duration_sec=$(echo "scale=4; $duration_ns / 1000000000" | bc)
throughput=$(echo "scale=2; $total_pairs / $duration_sec" | bc)

LOG_FILE="$BENCHMARK_DIR/results/cpu_throughput_${DATASET}_op${OPPOINT}.log"

{
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "Op Point: $OPPOINT"
    echo "Total Time: $duration_sec seconds"
    echo "Total Pairs: $total_pairs"
    echo "Throughput: $throughput pairs/second"
    echo "========================================"
} | tee "$LOG_FILE"

rm "$LIST_FILE"
