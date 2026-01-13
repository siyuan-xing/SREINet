import re
import numpy as np

# Read the log from a file
with open('training_log.txt', 'r') as file:
    log = file.read()

# Regular expressions to find the early stopping epoch, training loss, and total time elapsed
early_stopping_regex = re.compile(r"Early stopping at epoch (\d+)")
training_loss_regex = re.compile(r"Train Loss: ([0-9.e+-]+),")
time_elapsed_regex = re.compile(r"Time elapsed: ([0-9.]+)s")

# Lists to store extracted data
epochs_at_early_stopping = []
training_losses = []
times_elapsed = []

# Split log into different training dimensions
dimensions_logs = log.split("Training dimension")

print(f"Total number of dimension sections found: {len(dimensions_logs) - 1}")

# Track missing data
missing_early_stop = []
missing_loss = []
missing_time = []

for idx, dim_log in enumerate(dimensions_logs[1:], start=1):
    # Find the early stopping epoch
    early_stop_match = early_stopping_regex.search(dim_log)
    if early_stop_match:
        epochs_at_early_stopping.append(int(early_stop_match.group(1)))
    else:
        missing_early_stop.append(idx)
    
    # Find the last training loss before early stopping
    train_loss_matches = training_loss_regex.findall(dim_log)
    if train_loss_matches:
        training_losses.append(float(train_loss_matches[-1]))
    else:
        missing_loss.append(idx)

    # Find the time elapsed
    time_elapsed_match = time_elapsed_regex.search(dim_log)
    if time_elapsed_match:
        times_elapsed.append(float(time_elapsed_match.group(1)))
    else:
        missing_time.append(idx)

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

if training_losses:
    average_training_loss = np.mean(training_losses)
    std_training_loss = np.std(training_losses)
else:
    average_training_loss = 0
    std_training_loss = 0

if times_elapsed:
    average_time_elapsed = np.mean(times_elapsed)
    std_time_elapsed = np.std(times_elapsed)
else:
    average_time_elapsed = 0
    std_time_elapsed = 0

# Output the results
print(f"\n=== Results ===")
print(f"Average Early Stopping Epoch: {average_epoch:.2f} ± {std_epoch:.2f}")
print(f"Average Training Loss at Early Stopping: {average_training_loss:.12f} ± {std_training_loss:.12f}")
print(f"Average Time Elapsed: {average_time_elapsed:.2f}s ± {std_time_elapsed:.2f}s")
