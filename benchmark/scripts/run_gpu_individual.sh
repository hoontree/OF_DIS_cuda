#!/bin/bash
# Usage: ./run_gpu_individual.sh <dataset> <oppoint>

source "$(dirname "$0")/common.sh"

DATASET=$1
OPPOINT=${2:-2}

BUILD_DIR="$BENCHMARK_DIR/../build_cuda"
EXEC_NAME="run_OF_INT_CUDA"
OUT_DIR="$BENCHMARK_DIR/results/gpu_individual_${DATASET}_op${OPPOINT}"
CSV_FILE="$BENCHMARK_DIR/results/gpu_individual_${DATASET}_op${OPPOINT}.csv"
LIST_FILE="temp_pair_list_gpu_ind.txt"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$CSV_FILE")"
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

# Assign GPU: Op N -> GPU N-1
GPU_ID=$((OPPOINT - 1))
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running on GPU $GPU_ID for Op $OPPOINT..."
echo "image1,image2,time_total_ms,time_load_ms,time_pyramid_ms,time_oflow_ms,time_save_ms" > "$CSV_FILE"

total_pairs=$(wc -l < "$LIST_FILE")
echo "Processing $total_pairs pairs from $DATASET..."

while read img1 img2; do
    name1=$(basename "$img1" .png)
    name2=$(basename "$img2" .png)
    
    # Extract scene name (parent directory name)
    scene_dir=$(basename "$(dirname "$img1")")
    
    # Create scene subdirectory in output
    scene_out_dir="$OUT_DIR/$scene_dir"
    mkdir -p "$scene_out_dir"
    
    out_file="$scene_out_dir/${name1}_to_${name2}.flo"
    
    output=$("$BUILD_DIR/$EXEC_NAME" "$img1" "$img2" "$out_file" "$OPPOINT" 2>&1)
    
    if [[ "$output" == *"ERROR"* ]]; then
        echo "$img1,$img2,ERROR,0,0,0,0" >> "$CSV_FILE"
        continue
    fi

    # Parse timing
    time_load=$(echo "$output" | awk '/TIME \(Image loading/{print $NF; exit}' || echo "0")
    time_pyramid=$(echo "$output" | awk '/TIME \(Pyramide\+Gradients\)/{print $NF; exit}' || echo "0")
    time_oflow=$(echo "$output" | awk '/TIME \(O\.Flow Run-Time/{print $NF; exit}' || echo "0")
    time_save=$(echo "$output" | awk '/TIME \(Saving flow file/{print $NF; exit}' || echo "0")

    # Calculate total time
    time_total=$(awk -v a="$time_load" -v b="$time_pyramid" -v c="$time_oflow" -v d="$time_save" 'BEGIN{printf "%.6f", a+b+c+d}')
    
    echo "$img1,$img2,$time_total,$time_load,$time_pyramid,$time_oflow,$time_save" >> "$CSV_FILE"
    
done < "$LIST_FILE"

rm "$LIST_FILE"
echo "Done. Results saved to $CSV_FILE"
