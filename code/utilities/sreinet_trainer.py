import tensorflow as tf
import numpy as np
import time

"""SREINetTrainer module.

This module allows the user to create a SREINetTrainer.
    
    Author: Siyuan (Simon) Xing and Qingyu Han
    Email: sixing@calpoly.edu
    Licence: MIT Licence
    Copyright (c) 2025
    Version: 1.0.1 
"""


class SREINetTrainer:
    """A trainer class for SREINet models with pruning.
    
    This class encapsulates the training functionality for SREINet models,
    including pruning, early stopping, and callback support.
    """
    
    def __init__(self, model, optimizer, loss_fn, data_type='float32', random_seed=42):
        """Initialize the trainer.
        
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
        """Execute one training step."""
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
    
    def _prune_model(self, pruning_weight_threshold):
        """Apply pruning to the model weights."""
        binary_masks = []
        
        for var in self.model.trainable_variables:
            mask = tf.cast(tf.abs(var) >= pruning_weight_threshold, var.dtype)
            
            #The mask at position (0, 0) should be always 1 to avoid the constant term being pruned
            indices = tf.constant([[0, 0]])
            updates = tf.constant([1], dtype=mask.dtype)
            mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            
            binary_masks.append(mask)
            modified_weight = var * mask
            var.assign(modified_weight)
            
        return binary_masks
    
    def fit(self, training_input, training_output, batch_size, validation_split,
            epochs, pruning_scheduler, callbacks=None,
            early_stopping_loss_threshold=1e-6,
            early_stopping_pruning_threshold=1e-6,
            patience_periods=5, improvement_threshold=0.3,
            verbose=0, print_every_n_epochs=1, shuffle_per_epoch=True):
        """Train the model with pruning.
        
        Args:
            training_input: Input features for training
            training_output: Target outputs for training
            batch_size: Number of samples per batch
            validation_split: Fraction of data for validation
            epochs: Number of complete passes through the training dataset
            pruning_scheduler: PruningScheduler for gradually increasing sparsity
            callbacks: List of callbacks for custom actions
            early_stopping_loss_threshold: Threshold for early stopping based on loss
            early_stopping_pruning_threshold: Threshold for early stopping based on pruning
            patience_periods: Number of pruning periods to wait for improvement before early stopping (default: 3)
            improvement_threshold: Minimum relative improvement to reset patience counter (default: 0.1)
            verbose: Verbosity mode (0 = silent, 1 = progress bar)
            print_every_n_epochs: Print training progress every N epochs (default: 1)
            shuffle_per_epoch: Whether to shuffle training data at the beginning of each epoch (default: True)
            
        Returns:
            dict: Training history containing loss metrics
        """
        # Cast data to specified type
        training_input = tf.cast(training_input, self.data_type)
        training_output = tf.cast(training_output, self.data_type)
        
        # Prepare datasets
        total_samples = training_input.shape[0]

        # Create initial dataset with full shuffling for better training performance
        # Use full dataset size for initial shuffle to ensure complete randomization
        full_dataset = tf.data.Dataset.from_tensor_slices(
            (training_input, training_output)).shuffle(buffer_size=total_samples, seed=self.random_seed)
        
        train_samples = int(total_samples * (1 - validation_split))
        
        # Create train and validation datasets
        train_dataset = full_dataset.take(train_samples)
        val_dataset = full_dataset.skip(train_samples).batch(batch_size)
                
        # Initialize metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=self.data_type)
        val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=self.data_type)
        history = {'loss': [], 'val_loss': []}
        
        # Initialize patience tracking (period-based)
        best_val_loss = 999999 #initialize to a large number
        patience_counter = 0
        track_period = -1
        period_best_loss = 999999 #initialize to a large number

        # Initialize best weights to current model state
        self.best_weights = [var.copy() for var in self.model.get_weights()]
        self.best_loss = best_val_loss
        
        # Setup callbacks
        if callbacks:
            for callback in callbacks:
                callback.set_model(self.model)
                callback.params = {'epochs': epochs}
                callback.on_train_begin(logs={'loss_threshold': early_stopping_loss_threshold})
        
        training_start_time = time.time()
        
        # Training loop

        # Batch the training dataset
        epoch_train_dataset = train_dataset.batch(batch_size)

        for epoch in range(epochs):
            if callbacks:
                epoch_logs = {}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, logs=epoch_logs)
            
            if verbose and (epoch % print_every_n_epochs == 0 or epoch == epochs - 1):
                epoch_start_time = time.time()
            
            # Reset metrics
            train_loss.reset_state()
            val_loss.reset_state()
            
            # Apply pruning
            current_pruning_threshold = pruning_scheduler.get_threshold(epoch)
            binary_mask = self._prune_model(current_pruning_threshold)
            
            # Prepare training dataset for this epoch with shuffling and batching
            if shuffle_per_epoch:
                # Shuffle the training data for each epoch to get different mini-batch sequences
                # Use epoch-specific seed for reproducible but different shuffling each epoch
                epoch_seed = self.random_seed + epoch 
                epoch_train_dataset = train_dataset.shuffle(buffer_size=min(train_samples, 10000), seed=epoch_seed).batch(batch_size)
            
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

            # Update best weights if current loss is better
            if validation_split > 0:
                # Use validation loss for comparison when available
                current_loss_for_best = current_val_loss
            else:
                # Use training loss when no validation data
                current_loss_for_best = current_train_loss

            if current_loss_for_best < self.best_loss:
                self.best_loss = current_loss_for_best
                self.best_weights = [var.copy() for var in self.model.get_weights()]
            
            # Handle callbacks
            epoch_logs = {'loss': current_train_loss, 'val_loss': current_val_loss}
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs=epoch_logs)
            
            # Period-based patience tracking
            current_period = pruning_scheduler.get_period(epoch)
            
            # Check if we've moved to a new period
            if current_period != track_period:
                # New period started, check if previous period had improvement
                if track_period >= 0:  # Not the first period
                    if period_best_loss < best_val_loss:
                        # Calculate relative improvement
                        if best_val_loss > 1e-10:
                            relative_improvement = (best_val_loss - period_best_loss) / best_val_loss
                        else:
                            relative_improvement = 1.0  # Any improvement when loss is very small

                        if relative_improvement >= improvement_threshold:
                            # Significant improvement found, reset patience and save weights
                            best_val_loss = period_best_loss
                            patience_counter = 0
                            self.best_loss = period_best_loss
                            self.best_weights = [var.copy() for var in self.model.get_weights()]
                        else:
                            # Improvement is too small, increment patience
                            patience_counter += 1
                    else:
                        # No improvement, increment patience
                        patience_counter += 1
                
                # Start tracking new period
                track_period = current_period
                period_best_loss = current_val_loss
            else:
                # Same period, update period best loss
                period_best_loss = min(period_best_loss, current_val_loss)
            
            if verbose and (epoch % print_every_n_epochs == 0 or epoch == epochs - 1):
                epoch_time = time.time() - epoch_start_time
                if validation_split > 1e-10:
                    print(f"Epoch {epoch + 1:3.0f} (Period {current_period + 1}):"
                        f"  Train Loss: {current_train_loss:4.3e}"
                        f",  Validation Loss: {current_val_loss:4.3e}"
                        f",  Epoch Time: {epoch_time:.2f}s"
                        f",  Current Pruning Threshold: {current_pruning_threshold:.5g}")
                else:
                    print(f"Epoch {epoch + 1:3.0f} (Period {current_period + 1}):"
                        f"  Train Loss: {current_train_loss:4.3e}"
                        f",  Epoch Time: {epoch_time:.2f}s"
                        f",  Current Pruning Threshold: {current_pruning_threshold:.5g}")
                
            # Patience-based early stopping check
            if patience_counter >= patience_periods:
                print(f"Early stopping at epoch {epoch+1} due to patience limit reached. "
                      f"No significant improvement for {patience_periods} consecutive pruning periods.")
                break
            
            # Loss-based early stopping check
            if (current_train_loss < early_stopping_loss_threshold and 
                current_pruning_threshold >= early_stopping_pruning_threshold):
                if self.best_weights is None:
                    self.best_weights = [var.copy() for var in self.model.get_weights()]
                    self.best_loss = current_loss_for_best
                print(f"Early stopping at epoch {epoch+1} as train loss "
                      f"{current_train_loss:.14f} is below the threshold "
                      f"{early_stopping_loss_threshold:.14f}")
                break
        
        # Finalize training
        training_time = time.time() - training_start_time
        print(f'Training Stopped. Min loss: {self.best_loss:.14f}  Time elapsed: {training_time:.2f}s')

        # If training completed normally (not early stopped), save final weights as best if they're better
        if validation_split > 0:
            # Use validation loss for comparison
            final_loss = current_val_loss
        else:
            # Use training loss for comparison
            final_loss = current_train_loss

        if final_loss < self.best_loss:
            self.best_loss = final_loss
            self.best_weights = [var.copy() for var in self.model.get_weights()]
            print(f'Final model saved as best with loss: {final_loss:.6e}')

        if callbacks:
            for callback in callbacks:
                callback.on_train_end(logs={
                    'final_loss': current_train_loss,
                    'final_val_loss': current_val_loss
                })

        return history
    
    def get_best_weights(self):
        """Get the best weights saved during training.
        
        Returns:
            list: List of best weight arrays, or None if no weights have been saved
        """
        return self.best_weights
    
    def get_best_loss(self):
        """Get the best (minimum) validation loss achieved during training.
        
        Returns:
            float: Best validation loss, or infinity if no training has occurred
        """
        return self.best_loss


    def reset_trainer_and_model_weights(self, i):
        if hasattr(self.model.layers[1], 'reinitialize_weights'):
                tf.random.set_seed(self.random_seed + i)
                self.model.layers[1].reinitialize_weights()
        else:
            print("SREINet does not have a reinitialize_weights() function.")
        
        self.best_weights = None
        self.best_loss = float('inf')
    
