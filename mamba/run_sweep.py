import subprocess
import json
import os
from datetime import datetime

def run_experiment(input_seq_len, hidden_dim):
    # Create output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"mamba_models/sweep_{timestamp}_seq{input_seq_len}_dim{hidden_dim}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    # Initialize metrics file
    metrics_file = os.path.join(output_dir, "all_metrics.json")
    metrics_data = {
        "final_metrics": {},
        "all_evaluations": []
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    # Construct the command
    cmd = [
        "python", "train_mamba.py",
        "--num_examples", "100000",
        "--eval_steps", "100",
        "--input_seq_len", str(input_seq_len),
        "--num_kv_pairs", "8",
        "--epochs", "64",
        "--hidden_dim", str(hidden_dim),
        "--output_dir", output_dir,
        "--disable_wandb", "True"  # Disable wandb for sweep
    ]
    
    # Run the command
    print(f"\nRunning experiment with input_seq_len={input_seq_len}, hidden_dim={hidden_dim}")
    print(f"Command: {' '.join(cmd)}")
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env
    )
    
    # Collect output and print in real-time
    all_stdout = []
    all_stderr = []
    current_metrics = {}
    
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        
        if stdout_line:
            print(stdout_line.strip())
            all_stdout.append(stdout_line)
            
            # Check for metrics in real-time
            if "=== Evaluation at step" in stdout_line:
                # Start of new evaluation
                current_metrics = {}
            elif stdout_line.startswith("Metrics:"):
                try:
                    # Extract the metrics dictionary from the line
                    metrics_str = stdout_line.split("Metrics:", 1)[1].strip()
                    current_metrics = eval(metrics_str)  # Safely evaluate the string as a dict
                    
                    # Update metrics file in real-time
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)
                    
                    metrics_data["all_evaluations"].append(current_metrics)
                    metrics_data["final_metrics"] = current_metrics
                    
                    with open(metrics_file, "w") as f:
                        json.dump(metrics_data, f, indent=2)
                        
                except Exception as e:
                    print(f"Error parsing metrics: {e}")
                    continue
                    
        if stderr_line:
            print(stderr_line.strip())
            all_stderr.append(stderr_line)
            
        if stdout_line == '' and stderr_line == '' and process.poll() is not None:
            break
    
    stdout = ''.join(all_stdout)
    stderr = ''.join(all_stderr)
    
    # Save the output
    output_file = os.path.join(output_dir, "run_output.txt")
    with open(output_file, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)
    
    return current_metrics

def main():
    # Define parameter sweep values
    input_seq_lens = [128, 256, 512]
    hidden_dims = [128, 256]
    
    # Create results directory
    results_dir = "sweep_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nCreated results directory: {results_dir}")
    
    # Run all combinations
    all_results = {}
    for seq_len in input_seq_lens:
        for hidden_dim in hidden_dims:
            metrics = run_experiment(seq_len, hidden_dim)
            key = f"seq{seq_len}_dim{hidden_dim}"
            all_results[key] = metrics
    
    # Save all results to a single file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"sweep_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSweep completed. Results saved to {results_file}")

if __name__ == "__main__":
    main() 