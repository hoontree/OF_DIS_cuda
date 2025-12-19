#!/bin/bash
# Usage: ./visualize_flow.sh [directory]
# directory: Root directory to search for .flo files (default: results)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
COLOR_FLOW_EXEC="$BENCHMARK_DIR/../flow-code/color_flow"

SEARCH_DIR="${1:-$BENCHMARK_DIR/results}"

if [ ! -f "$COLOR_FLOW_EXEC" ]; then
    echo "Error: color_flow executable not found at $COLOR_FLOW_EXEC"
    exit 1
fi

if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR not found"
    exit 1
fi

echo "Searching for .flo files in $SEARCH_DIR..."

find "$SEARCH_DIR" -type f -name "*.flo" | while read flo_file; do
    dir_name=$(dirname "$flo_file")
    base_name=$(basename "$flo_file" .flo)
    out_png="$dir_name/${base_name}.png"
    
    # Check if png already exists to avoid re-processing (optional, but good for resume)
    # For now, we overwrite or just run.
    
    echo "Processing $flo_file -> $out_png"
    "$COLOR_FLOW_EXEC" "$flo_file" "$out_png"
done

echo "Visualization complete."
