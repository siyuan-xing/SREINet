import tensorflow as tf

class myLoss(tf.keras.losses.Loss):
    def __init__(self, model, loss, l1_regularization_strength=0.00, l2_regularization_strength=0.0, erf_scale=0.0):
        """
        Initializes the Loss class.

        Args:
            model: The model for which the loss will be calculated.
            loss: The loss function to be used. It can be a string representing a built-in loss function
                  or a custom loss function.
            l1_regularization_strength: The strength of L1 regularization. Default is 0.00.
            l2_regularization_strength: The strength of L2 regularization. Default is 0.0.
            erf_scale: The scale factor for the error function. Default is 0.0.

        Raises:
            ValueError: If the provided loss function is not supported.

        """
        super().__init__()
        self.model = model

        self.L1_REGULARIZATION_FACTOR = l1_regularization_strength
        self.L2_REGULARIZATION_FACTOR = l2_regularization_strength
        self.ERF_SCALE = erf_scale
        loss_functions = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mse_vec': self.loss_fn_MSE_vec,
            'lasso': self.loss_fn_Lasso,
            'ridge': self.loss_fn_Ridge,
            'lasso_erf': self.loss_fn_Lasso_with_erf,
            'elasticnet': self.loss_fn_ElasticNet
        }

        #if loss_name is a function, use it directly
        if callable(loss):
            self.loss_fn = loss
        elif loss.lower() not in loss_functions:
            raise ValueError(f"Loss function '{loss}' is not supported.")
        else:
            #make loss lower case
            self.loss_fn = loss_functions[loss.lower()]

    def call(self, y_true, y_pred):
        
        return self.loss_fn(y_true, y_pred)

    def loss_fn_MSE_vec(self, y_true, y_pred):

        return tf.reduce_mean(tf.square(y_true - y_pred),axis=0)

    def loss_fn_Lasso(self, y_true, y_pred):
        # Define the loss calculation without regularization
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Add regularization penalty 
        regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_variables])
        penalty = self.L1_REGULARIZATION_FACTOR * regularization_loss

        # Add penalty to the loss
        loss += penalty

        return loss

    def loss_fn_Ridge(self, y_true, y_pred):
        # Define the loss calculation without regularization
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Add regularization penalty (Ridge / L2 regularization)
        regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_variables])
        penalty = self.L2_REGULARIZATION_FACTOR * regularization_loss

        # Add penalty to the loss
        loss += penalty

        return loss
    
    def loss_fn_Lasso_with_erf(self, y_true, y_pred):
        # Define the loss calculation without regularization
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Apply erf to each weight and sum them up for regularization
        regularization_loss = tf.reduce_sum([
            tf.reduce_sum(tf.math.erf(tf.abs(self.ERF_SCALE*w))) for w in self.model.trainable_variables
        ])
        penalty = self.L1_REGULARIZATION_FACTOR * regularization_loss

        # Add penalty to the loss
        loss += penalty

        return loss
    

    #l0 is removed from elastic
    def loss_fn_ElasticNet(self, y_true, y_pred):
        """
        Loss function that includes L1, L2, and a conceptual L0 regularization penalties.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels.

        Returns:
        - Total loss including L1, L2, and conceptual L0 regularization penalties.
        """
        # Basic loss calculation without regularization
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Calculate L1 regularization loss
        l1_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_variables])
        
        # Calculate L2 regularization loss
        l2_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_variables])    

        # Apply L1, L2, and conceptual L0 regularization penalties
        penalty = (self.L1_REGULARIZATION_FACTOR * l1_regularization_loss) + \
                (self.L2_REGULARIZATION_FACTOR * l2_regularization_loss) 

        # Add penalties to the loss
        total_loss = loss + penalty

        return total_loss
