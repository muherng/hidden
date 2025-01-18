import subprocess
import matplotlib.pyplot as plt
import sys
import json

# Configuration
num_layers_range = [2,4,8,16]  # Example: Iterate over 1 to 10 layers
script_to_run = "transformer.py"  # Replace with the actual script name
results = []
model = 'RNN'

# Run the other script in a loop
for num_layers in num_layers_range:
    print(f"Running with num_layers={num_layers}")
    try: 
        # Use subprocess to run the other script
        result = subprocess.run(
            [sys.executable, script_to_run, "--data", "RNN", "--num_samples", "10000", "--input_dim", "100", f"--num_layers={num_layers}"],
            capture_output=True,
            text=True,
            check=True,
            env={"CUDA_VISIBLE_DEVICES": "2"}
        )
        results.append((num_layers, result.stdout))  # Store the result
    except subprocess.CalledProcessError as e:
        print(f"Error with num_layers={num_layers}: {e.stderr}")
        results.append((num_layers, None))  # Mark as failed

# Plot the results
# Define the values for args.num_layers
num_layers_list = [2, 4, 8, 16]
input_dim = 100

# Initialize dictionaries to store training and validation losses for each num_layers
training_losses = {}
validation_losses = {}

# Loop through each value of num_layers and load the corresponding loss data
for num_layers in num_layers_list:
    try:
        with open(f"./results/{input_dim}_{num_layers}losses.json", "r") as f:
            losses = json.load(f)

        # Store the losses in the dictionaries
        training_losses[num_layers] = losses.get("training_loss", [])
        validation_losses[num_layers] = losses.get("validation_loss", [])
    except FileNotFoundError:
        print(f"Warning: File for num_layers={num_layers} not found.")

# Plot training losses
plt.figure(figsize=(10, 6))
for num_layers, loss in training_losses.items():
    plt.plot(loss, label=f"Training Loss (num_layers={num_layers})", marker="o")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.legend()
plt.grid(True)
plt.savefig("./plots/training_losses_plot.png")
#plt.show()

# Plot validation losses
plt.figure(figsize=(10, 6))
for num_layers, loss in validation_losses.items():
    if loss:  # Only plot if validation losses exist
        plt.plot(loss, label=f"Validation Loss (num_layers={num_layers})", marker="x")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Validation Loss Over Steps")
plt.legend()
plt.grid(True)
plt.savefig("./plots/validation_losses_plot.png")
#plt.show()


