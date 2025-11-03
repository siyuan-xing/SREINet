"""SREINet module.

This module allows the user to create a SREINet.
    
    Author: Siyuan (Simon) Xing
    Email: sixing@calpoly.edu
    Licence: MIT Licence
    Copyright (c) 2025
    Version: 1.0.1 
"""

import tensorflow as tf
import numpy as np


class SREINet(tf.keras.Model):
    """SREINet.
        A NN class for the Sparse Regression Embedded Neural Network(SREINet). 
        The input data is splitted by its dimensions and fed sequentially to each layer.

        Attributes:
            layer_neuron_nums: the number of neurons in each hidden layer.
            output_layer_neuron_num: the number of neurons in the output layer. By default, it is 1.
            activations: the list of activation functions. The activation function will be applied to each dimension.
            weight_initializers: the dictionary of layer initializers.
            data_type: the data type of the layers.
            name: the name of the model.         
    """
    def __init__(self, layer_neuron_nums_per_act_fn, output_layer_neuron_num=1, activations=[lambda x:x], weight_initializer=tf.initializers.glorot_normal(), data_type='float32', name=None):
        """Constructor.
        
        Args:
            layer_neuron_nums_per_act_fn (list): the number of neurons per activation function in each hidden layer.
            output_layer_neuron_num (int): the number of neurons in the output layer. By default, it is 1.
            activations (list): the list of activation functions.
            weight_initializer (initializer): the dictionary of layer initializers. The user can specify the weight initializer of each layer.
            data_type (str): the data type of the layers.
            name (str): the name of the model.
        """
        super(SREINet, self).__init__(name=name)

        self._hidden_layer_num = len(layer_neuron_nums_per_act_fn)  # layer numbers
        #pre-processing
        my_weight_initializers = self.getLayerWeightsInitializer(weight_initializer)

        #create hidden and output layers
        self._hidden_layers = self.createHiddenLayers(layer_neuron_nums_per_act_fn, my_weight_initializers[:-1], data_type, activations)        
        if self._hidden_layer_num >1:
            self._output_layer = tf.keras.layers.Dense(output_layer_neuron_num, 
                                                    kernel_initializer = tf.initializers.ones(), 
                                                    use_bias=False, 
                                                    trainable=False) #output linear layer, not trainable, unit weights
        else:
            self._output_layer = tf.keras.layers.Dense(output_layer_neuron_num, 
                                                    kernel_initializer = tf.initializers.zeros(), 
                                                    use_bias=False, 
                                                    trainable=True)
            

    def call(self, inputs, training=False):

        xn = self._hidden_layers[0](inputs)
        for i in range(1, self._hidden_layer_num):
            xn = self._hidden_layers[i]([xn, inputs])
        outputs = self._output_layer(xn) 

        return outputs

    #utilities
    def createHiddenLayers(self, layer_units_per_act_fn,  weight_initializer_list, dtype, activations=[lambda x:x]):
        """
        Creates a list of hidden layers for the SREINet model.

        Args:
            layer_units_per_act_fn: the number of neurons in each hidden layer.
            weight_initializer_list: the list of weight initializers.
            dtype: the data type of the layers.
            activations: the list of activation functions.

        Returns:
            layers: the list of hidden layers.
        """
        layers = []

        layers.append(FirstHiddenLayer(data_type=dtype, activations=activations))

        for i in range(self._hidden_layer_num - 1):
            layers.append(SubsequentHiddenLayer(layer_units_per_act_fn[i + 1], 
                                              activations=activations,
                                               w_init=weight_initializer_list[i+1],
                                               data_type=dtype,)) 

        return layers

    def getLayerWeightsInitializer(self, w_initializer=tf.initializers.zeros()):
        total_layer_num = self._hidden_layer_num + 1


        #by default, last layer unit init weights, other layers zero init weights.
        weights_initializer = [w_initializer] * (total_layer_num)
        weights_initializer[-1] = tf.initializers.ones() # to be deprecated

        return weights_initializer

    def reinitialize_weights(self):
        """Reinitialize the weights of all layers."""
        for layer in self._hidden_layers:
            layer.reinitialize_weights()
        # Reset weights for the output layer if it's trainable
        if self._output_layer.trainable:
            self._output_layer.kernel.assign(self._output_layer.kernel_initializer(tf.shape(self._output_layer.kernel)))

class FirstHiddenLayer(tf.keras.layers.Layer):
    """The first hidden layers, which only accept one input.
    """
    def __init__(self, activations=[lambda x:x], data_type='float32'):
        """Constructor.
        The first hidden layer has no weights. 
        """
        super(FirstHiddenLayer, self).__init__()
        self.data_type = data_type
        self.activations = activations

    def call(self, inputs):
        """Forward-pass action for the first hidden layer.
        """
        combined_output = tf.ones((tf.shape(inputs)[0],1), dtype=self.data_type)
        
        # Apply each activation function and concatenate results
        for activation_func in self.activations:
            combined_output = tf.concat([combined_output, activation_func(inputs)], axis=1)
        return combined_output

    def reinitialize_weights(self):
        """Placeholder method for resetting weights. This layer has no weights."""
        pass  # No weights to reset


class SubsequentHiddenLayer(tf.keras.layers.Layer):
    """subsequent hidden layers. 

    Attributes:
        neuron_num: the number of neurons in the layer.
        data_type: the data type of the layer.
        activations: the list of activation functions.
    """
    def __init__(self, units_per_act_fn, activations=[lambda x:x], w_init = tf.initializers.zeros(), data_type='float32'):
        """Constructor.

        Args:
            units_per_act_fn (int): the number of neurons for each activate function. The total neuron number is units_per_act_fn * number_of_activation_functions + 1 (neuron with a constant one). 
            activations (list): the list of activation functions.
            w_init: the weight initializer.
            data_type (str): the data type of the layer.
        """
        super(SubsequentHiddenLayer, self).__init__()
        self.neuron_num = len(activations) * units_per_act_fn + 1 # one extra unit with value of 1
        self.data_type = data_type
        self.weight_initializer = w_init
        self.activations = activations
    
    def build(self, input_shape):
        """Build the layer.
        """
        shape1, shape2 = input_shape

        self.w = self.add_weight(shape=(shape1[1], self.neuron_num), 
                                 initializer=self.weight_initializer,
                                 trainable=True,
                                 name = 'w')
     
    def call(self, inputs):
        """Forward-pass action for the subsequent layers.
        """

        input, coord = inputs

        combined_output = tf.ones((tf.shape(input)[0],1), dtype=self.data_type)
        
        # Apply each activation function and concatenate results
        for activation_func in self.activations:
            combined_output = tf.concat([combined_output, activation_func(coord)], axis=1)

        return tf.matmul(input, self.w) * combined_output

    def reinitialize_weights(self):
        """Reinitialize the weights of the layer."""
        self.w.assign(self.weight_initializer(tf.shape(self.w)))


class LiftingLayer(tf.keras.layers.Layer):
    """
    Simple transformation layer that generates input to SREINet. This layer allows for incorporating 
    custom transformation functions such as 1/(ax+b) and xi*xj.
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            input_dim: Input dimension (number of input features)
            delta: Small constant to prevent division by zero
            seed: Random seed
        """
        super(LiftingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        indices_pairs = []
        for i in range(self.input_dim):
            for j in range(i, self.input_dim):
               indices_pairs.append([i, j])
        
        self.indices_pairs = tf.constant(indices_pairs, dtype=tf.int32)


    def call(self, inputs):
        """
        Apply transformations: [original_coords, 1/(ax+b), custom_transforms]
        """
        input_dim = inputs.shape[-1]
        tensor_prod = tf.einsum('bi,bj->bij', inputs, inputs) 
        

        B = tf.shape(tensor_prod)[0]  
        indices = tf.tile(self.indices_pairs[None, :, :], [B, 1, 1])  # (B, 15, 2)

        second_order = tf.gather_nd(tensor_prod, indices, batch_dims=1)

        sine_second_order = tf.sin(second_order)
    
        return tf.concat([inputs, sine_second_order], axis=-1)
