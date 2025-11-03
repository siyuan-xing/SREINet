import tensorflow as tf
import numpy as np
import time

class STLSQTrainer:
    """A trainer class implementing Sequential Thresholded Least Squares (STLSQ) for SREINet.

    This trainer alternates between:
    1. Training until loss plateaus
    2. Pruning weights below threshold
    3. Repeat until convergence (minimum loss of adjacent iterations are close)
    """

    def __init__(self, model, optimizer, loss_fn, data_type='float32', random_seed=42):
        """Initialize the STLSQ trainer.

        Args:
            model: SREINet model to be trained
            optimizer: Optimization algorithm for updating weights
            loss_fn: Loss function to measure model's performance
            data_type: Data type of the tensors (default: 'float32')
            random_seed: Random seed for shuffling operations (default: 42)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_type = data_type
        self.random_seed = random_seed
        self.best_weights = None
        self.best_loss = float('inf')

    @tf.function
    def _train_step(self, inputs, targets, binary_mask):
        """Execute one training step with masked gradients."""
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        masked_gradients = [tf.multiply(grad, mask) for grad, mask in zip(gradients, binary_mask)]
        self.optimizer.apply_gradients(zip(masked_gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def _val_step(self, inputs, targets):
        """Execute one validation step."""
        predictions = self.model(inputs, training=False)
        return self.loss_fn(targets, predictions)

    def _prune_weights(self, threshold, old_binary_masks):
        """Prune weights below threshold and return accumulated binary mask.

        Args:
            threshold: Threshold below which weights are pruned
            old_binary_masks: Previous binary masks to accumulate with new pruning

        Returns:
            list: Accumulated binary masks after applying new pruning
        """
        binary_masks = []

        for var, old_mask in zip(self.model.trainable_variables, old_binary_masks):
            # Create new mask based on current weights
            new_mask = tf.cast(tf.abs(var) >= threshold, var.dtype)

            # Keep constant term (position 0,0) unpruned
            indices = tf.constant([[0, 0]])
            updates = tf.constant([1], dtype=new_mask.dtype)
            new_mask = tf.tensor_scatter_nd_update(new_mask, indices, updates)

            # Accumulate masks: once pruned, always pruned
            accumulated_mask = tf.multiply(old_mask, new_mask)
            binary_masks.append(accumulated_mask)

            # Apply mask to set pruned weights to zero
            modified_weight = var * accumulated_mask
            var.assign(modified_weight)

        return binary_masks

    def _is_loss_plateau(self, loss_history, window_size=20, plateau_threshold=1e-6):
        """Check if loss has plateaued by examining recent history.

        Args:
            loss_history: List of recent loss values
            window_size: Number of epochs to consider for plateau detection
            plateau_threshold: Threshold for relative change to be considered flat

        Returns:
            bool: True if loss has plateaued
        """
        if len(loss_history) < window_size:
            return False

        recent_losses = loss_history[-window_size:]

        # Calculate relative change over the window
        if recent_losses[0] > 1e-10:
            relative_change = abs(recent_losses[-1] - recent_losses[0]) / recent_losses[0]
        else:
            relative_change = abs(recent_losses[-1] - recent_losses[0])

        return relative_change < plateau_threshold

    def fit(self, training_input, training_output, batch_size, validation_split,
            max_epochs_per_iteration=500, max_iterations=20,
            pruning_threshold=1e-3, plateau_window=20, plateau_threshold=1e-6,
            convergence_threshold=1e-5, callbacks=None,
            verbose=0, print_every_n_epochs=10, shuffle_per_epoch=True):
        """Train the model using STLSQ method.

        Args:
            training_input: Input features for training
            training_output: Target outputs for training
            batch_size: Number of samples per batch
            validation_split: Fraction of data for validation
            max_epochs_per_iteration: Maximum epochs for each training iteration
            max_iterations: Maximum number of prune-train iterations
            pruning_threshold: Threshold below which weights are pruned
            plateau_window: Number of epochs to check for plateau
            plateau_threshold: Threshold for relative loss change to be considered plateau
            convergence_threshold: Threshold for convergence between iterations
            callbacks: List of callbacks for custom actions
            verbose: Verbosity mode (0 = silent, 1 = detailed, 2 = very detailed)
            print_every_n_epochs: Print training progress every N epochs
            shuffle_per_epoch: Whether to shuffle training data each epoch

        Returns:
            dict: Training history containing loss metrics and iteration info
        """
        # Cast data to specified type
        training_input = tf.cast(training_input, self.data_type)
        training_output = tf.cast(training_output, self.data_type)

        # Prepare datasets
        total_samples = training_input.shape[0]

        # Create initial dataset with full shuffling
        full_dataset = tf.data.Dataset.from_tensor_slices(
            (training_input, training_output)).shuffle(buffer_size=total_samples, seed=self.random_seed)

        train_samples = int(total_samples * (1 - validation_split))

        # Create train and validation datasets
        train_dataset = full_dataset.take(train_samples)
        val_dataset = full_dataset.skip(train_samples)

        # Batch validation dataset
        val_dataset = val_dataset.batch(batch_size).map(
            lambda x, y: (tf.cast(x, self.data_type), tf.cast(y, self.data_type)))

        # Initialize metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=self.data_type)
        val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=self.data_type)

        # Initialize history
        history = {
            'loss': [],
            'val_loss': [],
            'iterations': []
        }

        # Initialize binary mask (no pruning initially)
        binary_mask = [tf.ones_like(var) for var in self.model.trainable_variables]

        # Initialize best weights
        self.best_weights = [var.copy() for var in self.model.get_weights()]
        self.best_loss = float('inf')

        # Setup callbacks
        if callbacks:
            for callback in callbacks:
                callback.set_model(self.model)
                callback.params = {'max_epochs': max_epochs_per_iteration * max_iterations}
                callback.on_train_begin(logs={})

        training_start_time = time.time()
        total_epochs = 0
        previous_min_loss = float('inf')

        # STLSQ iterations
        for iteration in range(max_iterations):
            if verbose >= 1:
                print(f"\n{'='*60}")
                print(f"STLSQ Iteration {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")

            iteration_loss_history = []
            iteration_start_epoch = total_epochs

            # Training phase: train until plateau
            for epoch in range(max_epochs_per_iteration):
                epoch_start_time = time.time()

                # Reset metrics
                train_loss.reset_state()
                val_loss.reset_state()

                # Prepare training dataset for this epoch
                if shuffle_per_epoch:
                    epoch_seed = self.random_seed + total_epochs
                    epoch_train_dataset = train_dataset.shuffle(
                        buffer_size=min(train_samples, 10000), seed=epoch_seed)
                else:
                    epoch_train_dataset = train_dataset

                # Batch and cast training dataset
                epoch_train_dataset = epoch_train_dataset.batch(batch_size).map(
                    lambda x, y: (tf.cast(x, self.data_type), tf.cast(y, self.data_type)))

                # Training
                for train_inputs, train_targets in epoch_train_dataset:
                    loss = self._train_step(train_inputs, train_targets, binary_mask)
                    train_loss(loss)

                # Validation
                for val_inputs, val_targets in val_dataset:
                    v_loss = self._val_step(val_inputs, val_targets)
                    val_loss(v_loss)

                # Record metrics
                current_train_loss = train_loss.result().numpy()
                current_val_loss = val_loss.result().numpy()
                history['loss'].append(current_train_loss)
                history['val_loss'].append(current_val_loss)
                iteration_loss_history.append(current_val_loss)

                # Update best weights
                if validation_split > 0:
                    current_loss_for_best = current_val_loss
                else:
                    current_loss_for_best = current_train_loss

                if current_loss_for_best < self.best_loss:
                    self.best_loss = current_loss_for_best
                    self.best_weights = [var.copy() for var in self.model.get_weights()]

                # Handle callbacks
                if callbacks:
                    epoch_logs = {'loss': current_train_loss, 'val_loss': current_val_loss}
                    for callback in callbacks:
                        callback.on_epoch_end(total_epochs, logs=epoch_logs)

                # Print progress
                if verbose >= 2 and (epoch % print_every_n_epochs == 0 or epoch == max_epochs_per_iteration - 1):
                    epoch_time = time.time() - epoch_start_time
                    print(f"  Epoch {epoch + 1:3.0f} (Global {total_epochs + 1:4.0f}):"
                          f"  Train Loss: {current_train_loss:4.3e}"
                          f",  Val Loss: {current_val_loss:4.3e}"
                          f",  Time: {epoch_time:.2f}s")

                total_epochs += 1

                # Check for plateau
                if self._is_loss_plateau(iteration_loss_history, plateau_window, plateau_threshold):
                    if verbose >= 1:
                        print(f"  Loss plateau detected at epoch {epoch + 1} (global epoch {total_epochs})")
                    break

            # Record iteration info
            iteration_min_loss = min(iteration_loss_history) if iteration_loss_history else float('inf')
            history['iterations'].append({
                'iteration': iteration + 1,
                'start_epoch': iteration_start_epoch,
                'end_epoch': total_epochs - 1,
                'min_loss': iteration_min_loss,
                'epochs_trained': len(iteration_loss_history)
            })

            if verbose >= 1:
                print(f"  Iteration {iteration + 1} complete: Min loss = {iteration_min_loss:.6e}")

            # Check convergence
            loss_change = abs(iteration_min_loss - previous_min_loss)
            if iteration > 0 and loss_change < convergence_threshold:
                if verbose >= 1:
                    print(f"\nConvergence achieved! Loss change ({loss_change:.6e}) < threshold ({convergence_threshold:.6e})")
                break

            previous_min_loss = iteration_min_loss

            # Pruning phase (except for last iteration)
            if iteration < max_iterations - 1:
                # Count weights before pruning
                total_weights_before = sum([tf.size(var).numpy() for var in self.model.trainable_variables])
                nonzero_before = sum([tf.math.count_nonzero(var).numpy() for var in self.model.trainable_variables])

                # Prune and accumulate mask (new pruning is added on top of previous masks)
                binary_mask = self._prune_weights(pruning_threshold, binary_mask)

                # Count weights after pruning
                nonzero_after = sum([tf.math.count_nonzero(var).numpy() for var in self.model.trainable_variables])

                if verbose >= 1:
                    sparsity = (1 - nonzero_after / total_weights_before) * 100
                    print(f"  Pruning: {nonzero_before} → {nonzero_after} nonzero weights "
                          f"({nonzero_before - nonzero_after} pruned, {sparsity:.1f}% sparse)")

        # Finalize training
        training_time = time.time() - training_start_time

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f'STLSQ Training Complete')
            print(f'Total iterations: {len(history["iterations"])}')
            print(f'Total epochs: {total_epochs}')
            print(f'Best loss: {self.best_loss:.6e}')
            print(f'Time elapsed: {training_time:.2f}s')
            print(f"{'='*60}\n")

        if callbacks:
            for callback in callbacks:
                callback.on_train_end(logs={
                    'final_loss': history['loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'best_loss': self.best_loss
                })

        return history

    def get_best_weights(self):
        """Get the best weights saved during training."""
        return self.best_weights

    def get_best_loss(self):
        """Get the best (minimum) validation loss achieved during training."""
        return self.best_loss

    def reset_trainer_and_model_weights(self, i):
        """Reset trainer state and reinitialize model weights."""
        if hasattr(self.model.layers[1], 'reinitialize_weights'):
            tf.random.set_seed(self.random_seed + i)
            self.model.layers[1].reinitialize_weights()
        else:
            print("SREINet does not have a reinitialize_weights() function.")

        self.best_weights = None
        self.best_loss = float('inf')
