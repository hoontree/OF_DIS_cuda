import os
import glob
import csv
import re

def parse_throughput_log(filepath):
    """Parses the throughput log file to extract the throughput value."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r'Throughput:\s+([\d\.]+)\s+pairs/second', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return 0.0

def clean_op_name(op_str):
    """Removes file extensions from op name."""
    return op_str.split('.')[0]

def generate_stats(results_dir):
    # Iterate over all files in the results directory
    # Expected filename patterns:
    # CSV: {mode}_individual_{dataset}_op{op}_t{t}.csv / {mode}_individual_{dataset}_op{op}.csv
    # Log: {mode}_throughput_{dataset}_op{op}.log

    # We'll group by (mode, dataset, op)
    # First, let's find all CSVs
    csv_files = glob.glob(os.path.join(results_dir, '*_individual_*.csv'))
    
    data_map = {} # Key: (mode, dataset, op), Value: {latency_metrics...}

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        parts = basename.split('_')
        # parts: [mode, individual, dataset, op...]
        
        if len(parts) < 4:
            continue
            
        mode = parts[0] # cpu / gpu
        dataset = parts[2] # kitti / sintel
        
        # Extract op point
        op_part_raw = [p for p in parts if p.startswith('op')]
        if not op_part_raw:
            continue
        op = clean_op_name(op_part_raw[0]) # op1, op2, ...
        
        # Parse CSV using csv module
        try:
            total_time = 0.0
            total_load = 0.0
            total_pyramid = 0.0
            total_oflow = 0.0
            total_save = 0.0
            count = 0
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        total_time += float(row['time_total_ms'])
                        total_load += float(row['time_load_ms'])
                        total_pyramid += float(row['time_pyramid_ms'])
                        total_oflow += float(row['time_oflow_ms'])
                        total_save += float(row['time_save_ms'])
                        count += 1
                    except ValueError:
                        continue # Skip invalid rows
            
            if count == 0:
                continue
                
            avg_total = total_time / count
            avg_load = total_load / count
            avg_pyramid = total_pyramid / count
            avg_oflow = total_oflow / count
            avg_save = total_save / count
            
            key = (mode, dataset, op)
            if key not in data_map:
                data_map[key] = {
                    'avg_total': avg_total,
                    'avg_load': avg_load,
                    'avg_pyramid': avg_pyramid,
                    'avg_oflow': avg_oflow,
                    'avg_save': avg_save,
                    'throughput': 0.0
                }
            else:
                pass

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    # Now look for throughput logs
    log_files = glob.glob(os.path.join(results_dir, '*_throughput_*.log'))
    for log_file in log_files:
        basename = os.path.basename(log_file)
        parts = basename.split('_')
        
        if len(parts) < 4:
            continue
            
        mode = parts[0]
        dataset = parts[2]
        
        op_part_raw = [p for p in parts if p.startswith('op')]
        if not op_part_raw:
            continue
        op = clean_op_name(op_part_raw[0])
        
        throughput = parse_throughput_log(log_file)
        
        key = (mode, dataset, op)
        if key in data_map:
            data_map[key]['throughput'] = throughput
        else:
             data_map[key] = {
                    'avg_total': 0, 'avg_load': 0, 'avg_pyramid': 0, 'avg_oflow': 0, 'avg_save': 0,
                    'throughput': throughput
                }

    # Format Output
    print(f"{'Mode':<5} | {'Dataset':<8} | {'Op':<4} | {'Throughput (pairs/s)':<22} | {'Avg Latency (ms)':<16} | {'Load':<8} | {'Pyramid':<8} | {'OFlow':<8} | {'Save':<8}")
    print("-" * 110)

    # Sort keys for consistent output
    sorted_keys = sorted(data_map.keys())
    
    output_csv_path = 'benchmark_summary.csv'
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Mode', 'Dataset', 'Op', 'Throughput', 'Avg_Total_ms', 'Avg_Load_ms', 'Avg_Pyramid_ms', 'Avg_OFlow_ms', 'Avg_Save_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted_keys:
            mode, dataset, op = key
            metrics = data_map[key]
            
            # Print to stdout
            print(f"{mode:<5} | {dataset:<8} | {op:<4} | {metrics['throughput']:<22.2f} | {metrics['avg_total']:<16.2f} | {metrics['avg_load']:<8.2f} | {metrics['avg_pyramid']:<8.2f} | {metrics['avg_oflow']:<8.2f} | {metrics['avg_save']:<8.2f}")
            
            # Write to CSV
            writer.writerow({
                'Mode': mode,
                'Dataset': dataset,
                'Op': op,
                'Throughput': metrics['throughput'],
                'Avg_Total_ms': metrics['avg_total'],
                'Avg_Load_ms': metrics['avg_load'],
                'Avg_Pyramid_ms': metrics['avg_pyramid'],
                'Avg_OFlow_ms': metrics['avg_oflow'],
                'Avg_Save_ms': metrics['avg_save']
            })
    
    print(f"\nStats exported to {os.path.abspath(output_csv_path)}")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if os.path.exists(results_dir):
        generate_stats(results_dir)
    else:
        print(f"Directory not found: {results_dir}")
