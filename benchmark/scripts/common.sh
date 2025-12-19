#!/bin/bash

# Base directories relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
SINTEL_ROOT="$BENCHMARK_DIR/Sintel/training"
KITTI_ROOT="$BENCHMARK_DIR/KITTI/training"

# Usage: collect_pairs_sintel <output_file>
collect_pairs_sintel() {
    local out_file="$1"
    
    local passes=("clean" "final" "albedo")
    for pass in "${passes[@]}"; do
        # Find all directories named $pass under SINTEL_ROOT
        find "$SINTEL_ROOT" -type d -name "$pass" | while read pass_dir; do
            # Recursively find all directories under this pass_dir
            find "$pass_dir" -type d | sort | while read seq_dir; do
                # Find pngs in this directory (maxdepth 1 to avoid duplicates from subdirs)
                frames=($(find "$seq_dir" -maxdepth 1 -type f -name '*.png' | sort))
                count=${#frames[@]}
                if [ "$count" -ge 2 ]; then
                    for ((i=0; i<count-1; i++)); do
                        echo "${frames[i]} ${frames[i+1]}" >> "$out_file"
                    done
                fi
            done
        done
    done
}

# Usage: collect_pairs_kitti <output_file>
collect_pairs_kitti() {
    local out_file="$1"
    
    local subdirs=("image_2" "image_3")
    for subdir in "${subdirs[@]}"; do
        local search_dir="$KITTI_ROOT/$subdir"
        if [ -d "$search_dir" ]; then
            # Recursively find all _10.png files
            find "$search_dir" -name "*_10.png" | sort | while read img1; do
                img2="${img1/_10.png/_11.png}"
                if [ -f "$img2" ]; then
                    echo "$img1 $img2" >> "$out_file"
                fi
            done
        fi
    done
}
