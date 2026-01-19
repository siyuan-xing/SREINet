import re
import numpy as np
import os

# Read the log from a file
log_file = 'training_log.txt'
if not os.path.exists(log_file):
    print(f"Error: File '{log_file}' not found!")
    exit(1)

with open(log_file, 'r', encoding='utf-8') as file:
    log = file.read()

if not log.strip():
    print(f"Error: File '{log_file}' is empty!")
    exit(1)

# Regular expressions to find the early stopping epoch, training loss, and total time elapsed
# Extract epoch and loss from "Early stopping at epoch X as train loss Y is below..."
early_stopping_regex = re.compile(r"Early stopping at epoch (\d+)")
early_stopping_with_loss_regex = re.compile(r"Early stopping at epoch (\d+) as train loss ([0-9.e+-]+)")
training_loss_regex = re.compile(r"Train Loss: ([0-9.e+-]+),")
time_elapsed_regex = re.compile(r"Time elapsed: ([0-9.]+)s")

# Lists to store extracted data
epochs_at_early_stopping = []
training_losses = []
times_elapsed = []

# Split log into different training dimensions
# Handle both "Training dimension" and "Training dimension X" patterns
dimensions_logs = re.split(r"Training dimension \d+", log)
if len(dimensions_logs) == 1:
    # Try alternative split if the above didn't work
    dimensions_logs = log.split("Training dimension")

# If we only have one part and it contains training data, treat the whole log as one dimension
if len(dimensions_logs) == 1 and ("Early stopping" in log or "Train Loss" in log):
    dimensions_logs = [log]
    print("Note: Treating entire log as a single dimension")
elif len(dimensions_logs) > 1:
    # Remove the first part if it's just header info (before first "Training dimension")
    if dimensions_logs[0].strip() and not any(keyword in dimensions_logs[0] for keyword in ["Early stopping", "Train Loss", "Time elapsed"]):
        dimensions_logs = dimensions_logs[1:]
    else:
        dimensions_logs = dimensions_logs[1:] if len(dimensions_logs) > 1 else [log]
    
print(f"Total number of dimension sections found: {len(dimensions_logs)}")

# Track missing data
missing_early_stop = []
missing_loss = []
missing_time = []

for idx, dim_log in enumerate(dimensions_logs, start=1):
    # Find the early stopping epoch
    early_stop_match = early_stopping_regex.search(dim_log)
    if early_stop_match:
        epochs_at_early_stopping.append(int(early_stop_match.group(1)))
    else:
        missing_early_stop.append(idx)
        # Debug: print a snippet if not found
        if idx <= 3:
            print(f"Debug dimension {idx}: Early stopping not found. First 200 chars: {dim_log[:200]}")
    
    # Find the last training loss before early stopping
    train_loss_matches = training_loss_regex.findall(dim_log)
    if train_loss_matches:
        training_losses.append(float(train_loss_matches[-1]))
    else:
        missing_loss.append(idx)
        # Debug: print a snippet if not found
        if idx <= 3:
            print(f"Debug dimension {idx}: Training loss not found. First 200 chars: {dim_log[:200]}")

    # Find the time elapsed
    time_elapsed_match = time_elapsed_regex.search(dim_log)
    if time_elapsed_match:
        times_elapsed.append(float(time_elapsed_match.group(1)))
    else:
        missing_time.append(idx)
        # Debug: print a snippet if not found
        if idx <= 3:
            print(f"Debug dimension {idx}: Time elapsed not found. First 200 chars: {dim_log[:200]}")

# Print statistics
print(f"\nExtracted data counts:")
print(f"  Early stopping epochs: {len(epochs_at_early_stopping)}")
print(f"  Training losses: {len(training_losses)}")
print(f"  Time elapsed: {len(times_elapsed)}")

if missing_early_stop:
    print(f"\nWarning: Missing early stopping data for dimensions: {missing_early_stop[:10]}{'...' if len(missing_early_stop) > 10 else ''}")
if missing_loss:
    print(f"Warning: Missing training loss data for dimensions: {missing_loss[:10]}{'...' if len(missing_loss) > 10 else ''}")
if missing_time:
    print(f"Warning: Missing time elapsed data for dimensions: {missing_time[:10]}{'...' if len(missing_time) > 10 else ''}")

# Calculate average values only if we have data
if epochs_at_early_stopping:
    average_epoch = np.mean(epochs_at_early_stopping)
    std_epoch = np.std(epochs_at_early_stopping)
else:
    average_epoch = 0
    std_epoch = 0
    print("\nError: No early stopping epochs found!")

if training_losses:
    average_training_loss = np.mean(training_losses)
    std_training_loss = np.std(training_losses)
else:
    average_training_loss = 0
    std_training_loss = 0
    print("Error: No training losses found!")

if times_elapsed:
    average_time_elapsed = np.mean(times_elapsed)
    std_time_elapsed = np.std(times_elapsed)
else:
    average_time_elapsed = 0
    std_time_elapsed = 0
    print("Error: No time elapsed data found!")

# Output the results
if epochs_at_early_stopping and training_losses and times_elapsed:
    print(f"\n=== Results ===")
    if len(epochs_at_early_stopping) > 1:
        print(f"Average Early Stopping Epoch: {average_epoch:.2f} ± {std_epoch:.2f}")
        print(f"Average Training Loss at Early Stopping: {average_training_loss:.12f} ± {std_training_loss:.12f}")
        print(f"Average Time Elapsed: {average_time_elapsed:.2f}s ± {std_time_elapsed:.2f}s")
    else:
        print(f"Early Stopping Epoch: {epochs_at_early_stopping[0]}")
        print(f"Training Loss at Early Stopping: {training_losses[0]:.12f}")
        print(f"Time Elapsed: {times_elapsed[0]:.2f}s")
