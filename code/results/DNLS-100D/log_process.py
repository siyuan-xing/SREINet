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

for dim_log in dimensions_logs[1:]:
    # Find the early stopping epoch
    early_stop_match = early_stopping_regex.search(dim_log)
    if early_stop_match:
        epochs_at_early_stopping.append(int(early_stop_match.group(1)))
    
    # Find the last training loss before early stopping
    train_loss_matches = training_loss_regex.findall(dim_log)
    if train_loss_matches:
        training_losses.append(float(train_loss_matches[-1]))

    # Find the time elapsed
    time_elapsed_match = time_elapsed_regex.search(dim_log)
    if time_elapsed_match:
        times_elapsed.append(float(time_elapsed_match.group(1)))

# Calculate average values
average_epoch = np.mean(epochs_at_early_stopping)
average_training_loss = np.mean(training_losses)
average_time_elapsed = np.mean(times_elapsed)

# Output the results
print(f"Average Early Stopping Epoch: {average_epoch:.2f}")
print(f"Average Training Loss at Early Stopping: {average_training_loss:.13f}")
print(f"Average Time Elapsed: {average_time_elapsed:.2f}s")
