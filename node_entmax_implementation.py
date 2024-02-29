# -----------------------------------------------------------
# Dissertation Project: An Empirical Study on the Classification 
# Performance of Deep Learning vs. Gradient Boosting 
# on heterogeneous tabular data
#
# This python file is an adaptation of the entmax-15 function, 
# 
# Acknowledgments: adapted from the code: 
# https://gist.github.com/BenjaminWegener. 
# Currently there is no import for entmax-15 in tensorflow,
# only in pytorch.
# 
# Originally, the Entmax 1.5 implementation, is inspired by
#  paper: https://arxiv.org/pdf/1905.05702.pdf and
#  pytorch code: https://github.com/deep-spin/entmax
# 
# Author: Adam Mabrouk
# Supervisor: Ben Ralph
# Institution: University of Bath
# Created on: 01/01/2024
# Version: 1.0
# -----------------------------------------------------------
# Library versions
# ----------------------
# Python version: 3.11.5 
# tensorflow version: 2.15.0

import tensorflow as tf

def entmax15(inputs, axis=-1):
    """ The Entmax 1.5 activation function is a generalization of softmax,
    that allows for sparser distributions, '1.5' refering to the alpha parameter
    in the generalized Entmax function.
    
    Args:
        Tensor.
    Returns:
        output of the Entmax activation function. """
    
    def entmax_inner(inputs):
        """This function is for the activation of entmax 1.5, 
        Args:
            Tensor, processed through the entmax function
        Returns:
                Activated output and lambda function"""
        inputs_adjusted = adjust_input_for_stability(inputs, axis)
        threshold = compute_entmax_threshold(inputs_adjusted, axis)
        outputs = calculate_entmax_output(inputs_adjusted, threshold)
        return outputs, lambda loss_grad_wrt_outputs: entmax_gradient(
            loss_grad_wrt_outputs, outputs)

    return tf.custom_gradient(entmax_inner)(inputs)

def adjust_input_for_stability(inputs, axis):
    """ This function adjusts the input for numerical stability, 
    by halving and subtracting the maximum value, which helps to 
    prevent overflow (value too large)/underflow (value to small).
    Args:
        Tensor.
    Returns:
        Adjusted inputs. """
    inputs_halved = inputs / 2
    max_input = tf.reduce_max(inputs_halved, axis, keepdims=True)
    return inputs_halved - max_input

def compute_entmax_threshold(inputs, axis):
    """ This function calculates the threshold value (tau_star) used in Entmax activation.
        Threshold is used for clipping the inputs in the Entmax function.
    Args:
        Tensor.
    Returns:
        Threshold value.
    """
    tau_star, _ = entmax_threshold_and_support(inputs, axis)
    return tau_star

def calculate_entmax_output(inputs, threshold):
    """ Function calculates the output of the entmax activation.
    Args:
        Tensor, threshold value.
    Returns:
        Entmax activation function."""
    outputs_sqrt = tf.nn.relu(inputs - threshold)
    return tf.square(outputs_sqrt)

def entmax_gradient(loss_grad_wrt_outputs, outputs):
    """ This function calculates the gradients needed, 
    (loss gradients with respect to output) for backpropagation in NODE.
    Args:
        loss_grad_wrt_outputs, of the activation function.
    Returns:
        Gradient of inputs. """
    outputs_sqrt = tf.sqrt(outputs)
    scaled_loss_grad = loss_grad_wrt_outputs * outputs_sqrt
    sum_scaled_loss_grad = tf.reduce_sum(scaled_loss_grad, axis=-1, keepdims=True)
    sum_outputs_sqrt = tf.reduce_sum(outputs_sqrt, axis=-1, keepdims=True)
    q = sum_scaled_loss_grad / sum_outputs_sqrt
    return scaled_loss_grad - q * outputs_sqrt

def k_largest_entries(inputs, k, axis=-1, **kwargs):
    """
    This function finds the values of the 'k' largest entries for the last dimension.
    Args:
        Tensor.
        K largest entries
        **kwargs: for tf.nn.top_k.
    Returns:
        Tuple of tensors containing the top k values."""
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        permute_transpose = list(range(inputs.shape.num_dimensions))
        permute_transpose.append(permute_transpose.pop(axis))
        inverse_permute = [permute_transpose.index(i) for i in range(len(permute_transpose))]

        input_perm = tf.transpose(inputs, permute_transpose)
        sorted_transposed_tensor, sorted_indices_transposed = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        sorted_original_tensor = tf.transpose(sorted_transposed_tensor, inverse_permute)
        sorted_original_indicies = tf.transpose(sorted_indices_transposed, inverse_permute)
    return sorted_original_tensor, sorted_original_indicies


def generate_index_tensor(inputs, axis=-1):
    """This function maps positional indices to the structure of the input tensor, 
    making the shape of the output index tensor identical to that of the input tensor
    Args:
        Tensor.
    Returns:
        Tensor of indices, same shap as inputs"""
    assert inputs.shape.num_dimensions is not None, "Input tensor has a defined number of dimensions."
    index_range = tf.range(1, tf.shape(inputs)[axis] + 1, dtype=inputs.dtype)
    target_shape = [1] * inputs.shape.num_dimensions
    target_shape[axis] = -1  
    return tf.reshape(index_range, target_shape)

def gather_over_axis(values, indices, gather_axis):
    """ This function enables elements to be selected from a tensor, such as that seen in 
    advanced indexing, slicing. Provies useful functionality when working with TensorFlow 
    i.e., for example, versions <= 1.8, where `tf.gather` lacks certain capabilities.
    Args:
        Tensor of gathered values, same number of dimensions as 'values'
    Returns:
        Tensor with the same shape as `values`, containing the gathered elements. """
    assert indices.shape.num_dimensions is not None and values.shape.num_dimensions is not None
    assert indices.shape.num_dimensions == values.shape.num_dimensions

    num_dimensions = indices.shape.num_dimensions
    gather_axis = gather_axis % num_dimensions
    shape = tf.shape(indices)
    selectors = [indices if axis_i == gather_axis else 
                 tf.tile(tf.reshape(tf.range(shape[axis_i], dtype=indices.dtype), 
                                    [-1 if i == axis_i else 1 for i in range(num_dimensions)]),
                         [shape[i] if i != axis_i else 1 for i in range(num_dimensions)])
                 for axis_i in range(num_dimensions)]
    return tf.gather_nd(values, tf.stack(selectors, axis=-1))

def entmax_threshold_and_support(inputs, axis=-1):
    """ This function calculates the clipping threshold and support 
    size for each instance in the input tensor. 
    Args:
        Tensor containing modified inputs ((entmax1.5 inputs - max) / 2)) to entmax
    Returns:
        tau_star, Tensor with clipping threshold for each instance.
    """

    with tf.name_scope('entmax_threshold_and_support'):
        
        count_of_outcomes = tf.shape(inputs)[axis]
        sorted_inputs, _ = tf.nn.top_k(inputs, k=count_of_outcomes, sorted=True)
        sequence = tf.range(1, count_of_outcomes + 1, dtype=tf.float32)
        cumulative_sum_inputs = tf.cumulative_sum(sorted_inputs, axis=axis)
        average = cumulative_sum_inputs / sequence
        squared_average = tf.cumulative_sum(tf.square(sorted_inputs), axis=axis) / sequence
        deviation = (1 - sequence * (squared_average - tf.square(average))) / sequence
        positive_deviation = tf.nn.relu(delta)
        tau = average - tf.sqrt(positive_deviation)
        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, sorted_inputs), tf.int64), axis=axis, keepdims=True)
        tau_star = tf.gather_nd(tau, support_size - 1, axis)

    return tau_star, support_size