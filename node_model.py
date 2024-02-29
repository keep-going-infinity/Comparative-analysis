# -----------------------------------------------------------
# Dissertation Project: An Empirical Study on the Classification
# Performance of Deep Learning vs. Gradient Boosting
# on heterogeneous tabular data
#
# This python file contains the NODE architecture
#
# Author: Adam Mabrouk
# Supervisor: Ben Ralph
# Institution: University of Bath
# Created on: 01/01/2024
# Version: 1.0
#
# Acknowledgments:
# The NODE architecture in this code has been adapted from the Authors:
# Popov, S., Morozov, S. and Babenko, A., 2019. Neural oblivious decision ensembles for deep learning 
# on tabular data.
# arXiv preprint arXiv:1909.06312. Source paper: https://arxiv.org/pdf/1909.06312v2.pdf
# Adapted NODE architecture: Author: Sergey Popov, Source: https://github.com/Qwicen/node/blob/master/lib/odst.py
# Adapted Model training: Author: https://github.com/anonICLR2020/node/blob/master/lib/trainer.py
# -----------------------------------------------------------
# Library Versions Used
# ----------------------

# Python version: 3.11.5
# tensorflow: 2.15.0
# tensorflow_addons: 0.23.0
# tensorflow_probability: 0.23.0

# TensorFlow imports
from typing import Optional
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from node_entmax_implementation import entmax15


def create_node(params, X_train):
    """ This function compiles a Neural Oblivious Decision Ensembles (NODE) model for binary classification.
    The model's configuration is based on the optuna parameters (below),  uses sigmoid activation, and is compiled 
    with the Adam optimizer, AUC metric, and binary crossentropy loss for binary classification.

    Params:
        Dictionary using the following parameters: 'soft_tree_layer', 'neurons', 'dropout_rate',
        'depth_of_tree', 'soft_trees', and 'learning_rate'.
        
        X_train: Input training data,

    Returns:
        Model: Compiled/configured NODE model. """

    link = tf.keras.activations.sigmoid

    soft_tree_layer = int(params.get("soft_tree_layer", 1))
    dropout_rate = float(params.get("dropout_rate", 0.3))
    depth_of_tree = int(params.get("depth_of_tree", 4))
    soft_trees = int(params.get("soft_trees", 3))

    model = NODE(soft_tree_layer=soft_tree_layer, neurons=1,
                 dropout_rate=dropout_rate,
                 depth_of_tree=depth_of_tree,
                 soft_trees=soft_trees,
                 link=link)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                  metrics=[tf.keras.metrics.AUC(name='auprc', curve='PR')],
                  loss='binary_crossentropy')
    return model


def optuna_node_params(trial):
    """ Generates a dict of hyperparameters for NODE using Optuna optimisation. 
    Params: Optuna optimises the params; number of layers, dropout rate, depth of tree, 
    number of soft trees, learning rate, batch size, and learning rate.

    Params:
        trial (optuna.trial.Trial): Optuna uses 'trial', an object to suggest hyperparameters.

    Returns:
        dict: A dictionary with suggested hyperparameters for the NODE model. """

    return {
        "soft_tree_layer": trial.suggest_int('soft_tree_layer', 2, 7),
        "dropout_rate": trial.suggest_float('dropout_rate', 0.1, 0.5),
        "depth_of_tree": trial.suggest_int('depth_of_tree', 2, 8),
        "soft_trees": trial.suggest_int('soft_trees', 1, 5),
        "seed": trial.suggest_int('seed', 1, 1000),  # or SEED
        "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        "lr_reduce_factor": trial.suggest_float("lr_reduce_factor", 0.1, 0.4),
        "lr_reduce_patience": trial.suggest_int("lr_reduce_patience", 7, 10),
        "lr_reduce_threshold": trial.suggest_float("lr_reduce_threshold", 1e-5, 1e-1, log=True)}


class ODT(tf.keras.layers.Layer):
    """ The Oblivious Decision Tree (ODT) class is a layer for the NODE architecture.
    The layer implements a single decision tree with soft decisions at each node. """

    def __init__(self, soft_trees: int = 6, depth_of_tree: int = 1, neurons: int = 1,
                 initialise_beta: float = 1.0, **kwargs):
        """ Params:
                soft_trees: Number of soft trees in the ODT layer.
                depth_of_tree: The depth of each tree.
                neurons: Number of output neurons for each tree.
                initialise_beta: Hyperparameter for initialising feature thresholds."""
        super(ODT, self).__init__()
        self.initialized = False
        self.soft_trees = soft_trees
        self.depth_of_tree = depth_of_tree
        self.neurons = neurons
        self.initialise_beta = initialise_beta

    @staticmethod
    def sparsemoid(inputs: tf.Tensor):
        """ Sparsemoid activation function. Provides a smooth and differentiable 
        approximation of the step function used in decision trees. The function transforms 
        input linearly, and after scaling, clips the result to the range [0, 1], ensuring 
        the output is always bounded, similar to a traditional sigmoid function.
        
        Params:
            tf.Tensor: Input tensor for the activation function.

        Returns:
            tf.Tensor: The output tensor, after applying the sparsemoid function, with values 
            clipped to the ranges mentioned above (0, 1). """

        if not isinstance(inputs, tf.Tensor):
            raise ValueError("Input must be a TensorFlow tensor")

        scaled = tf.multiply(inputs, 0.5)
        shifted = tf.add(scaled, 0.5)
        output = tf.clip_by_value(shifted, 0.0, 1.0)
        return output

    def compute_node_onehot_vectors(self, depth_of_tree):
        """ Compute one-hot vectors for tree nodes, which is based on the depth of the tree.

        Params:
            depth_of_tree (int): The depth of the tree, determins the size of the one-hot encoded vectors.

        Returns:
          tf.Tensor: float32 tensor containing one-hot vectors for each node in the tree. Shape is 
          [2, number_of_nodes, 2], where number_of_nodes equals 2^depth_of_tree. The last dimension 
          contains the one-hot encoded binary representation of the node's index. """
        
        node_indices = tf.keras.backend.arange(0, 2 ** depth_of_tree, 1)
        binary_offsets = 2 ** tf.keras.backend.arange(0, depth_of_tree, 1)
        binary_codes = (tf.reshape(node_indices, (1, -1)) // tf.reshape(binary_offsets, (-1, 1)) % 2)
        node_onehot_vectors = tf.stack([binary_codes, 1 - binary_codes], axis=-1)
        return tf.cast(node_onehot_vectors, 'float32')

    def initialise_response(self, soft_trees, neurons, depth_of_tree):
        
        """ Initialises the response variable for the layer with a shape tailored to the number of soft trees,
        neurons, and the tree depth.
        Params:
             soft_trees (int): Number of soft decision trees in the layer.
             neurons (int): Number of neurons in the layer.
             depth_of_tree (int): Depth of each tree.
        Returns:
             tf.Variable: Trainable variable initialised with 1's, shaped to accommodate the layer's
             architecture. Its shape is [soft_trees, neurons, 2^depth_of_tree], dtype float32."""
            
        response_init = tf.ones_initializer()
        return tf.Variable(initial_value=response_init(
            shape=(soft_trees,neurons,2 ** depth_of_tree),dtype='float32'), trainable=True)

    def initialise_soft_feature_scores(self, input_shape):
        """ Initialises the soft feature scores for the input features across all trees and their respective depths.

        Params:
            input_shape (Tuple[int,]): Shape of input layer, where the last dimension specifies the number of features.

        Returns:
            tf.Variable: Trainable weight variable for 'soft feature scores', 
            shape: [number_of_features, soft_trees, depth_of_tree], initialized to zeros. """
        
        soft_feature_parameters = tf.zeros_initializer()
        return self.add_weight(
            name="soft_feature_scores",
            shape=(input_shape[-1], self.soft_trees, self.depth_of_tree),
            initializer=soft_feature_parameters,
            trainable=True)

    def initialise_node_split_thresholds(self):
        """Initialises the node split thresholds for splitting nodes in the DTs.
           The thresholds are used to decide at which point to split a node based on the input features
           Returns:
               tf.Variable representing the trainable weights for node split thresholds.
               Includes: number of soft trees and their depth, allowing for unique thresholds at each decision point in the trees."""
        
        node_decision_boundaries = tf.zeros_initializer()
        return self.add_weight(
            name="node_split_thresholds",
            shape=(self.soft_trees, self.depth_of_tree),
            initializer=node_decision_boundaries,
            trainable=True)

    def initialise_soft_decision_temperature(self):
        """ Initialises the temperature for soft decisions in the trees. The term 'temperature' controls the 
        smoothness of the decision boundaries. Higher temperature results in smoother (softer) decisions, 
        while a lower temperature results in more discrete decisions. The introduction of smoother 'temperature'
        reduces stricter thresholds, instead instroducing gradients where the probability of belonging to one class 
        or another changes more smoothly. This allows NODE to express uncertainty in a more unique way, which is
        especially relevant in heterogeneous data, where within regions of the feature space, the data points from
        different classes are closely mixed together.
        Returns:
            tf.Variable representing the trainable weights for the soft decision temperature.
            Includes: number of soft trees and their depth, enabling unique temperature settings for each 
            decision point. """
        
        soft_logarithm = tf.ones_initializer()
        return self.add_weight(
            name="soft_decision_temperature",
            shape=(self.soft_trees, self.depth_of_tree),
            initializer=soft_logarithm,
            trainable=True)

    def build(self, input_shape: tf.TensorShape):
        """ Build the internal components of the ODT layer. Sets up the internal structure of the 
        Oblivious Decision Tree (ODT) layer by initialising the necessary components for the layer's operation.

        Params:
            input_shape (tf.TensorShape): Determines shape of the soft feature scores.

        The method does not return a value but initialises internal variables and tf.variables for the layer's 
        operation. This includes: soft feature scores, node split thresholds, soft decision temperature, node onehot vectors, 
        and response weights. """
        
        self.soft_feature_scores = self.initialise_soft_feature_scores(input_shape)
        self.node_split_thresholds = self.initialise_node_split_thresholds()
        self.soft_decision_temperature = self.initialise_soft_decision_temperature()
        self.node_onehot_vectors = tf.Variable(initial_value=self.compute_node_onehot_vectors(self.depth_of_tree),
                                               trainable=False)

        self.response = self.initialise_response(self.soft_trees, self.neurons, self.depth_of_tree)

    def initialize(self, inputs):
        """ Initialises model parameters.
        Params:
            inputs (tf.Tensor): Input tensor.
        Initialises node split thresholds and soft decision temperature (which referes to smoothness of decision boundaries)."""
        
        feature_values = self.feature_values(inputs)
        self.update_node_split_thresholds(feature_values)
        self.update_soft_decision_temperature(feature_values)

    def update_node_split_thresholds(self, feature_values): # need to check
        """ Initialises the thresholds for node splitting in the DTs.

        Params:
            feature_values (tf.Tensor): Tensor of feature values.
            The function calculates initial thresholds for node splits based on the percent of feature values.
            The 'quantile values' parameter is used to determine the thresholds for splitting nodes in the DTs """
        
        quantile_values = self.calculate_percentiles()
        flattened_feature_values = self.flatten_feature_values(feature_values)
        init_thresholds = self.calculate_initial_thresholds(flattened_feature_values, quantile_values)
        self.node_split_thresholds.assign(tf.reshape(init_thresholds, self.node_split_thresholds.shape))

    def calculate_percentiles(self):
        """ This function calculates the percentiles for initialising node split thresholds using the Beta 
            distribution.
        Returns:
            tf.Tensor: Tensor of percentile values derived from the beta distribution.
            The function enables a balanced distribution of percentiles across the DTs. """
        
        beta_distribution = tfp.distributions.Beta(self.initialise_beta, self.initialise_beta)
        return 100 * beta_distribution.sample([self.soft_trees * self.depth_of_tree])

    def flatten_feature_values(self, feature_values):
        """Flattens the feature values to a 2D tensor.
        Params:
            feature_values to be flattened (tf.Tensor).
        Returns:
            tf.Tensor: Flattened feature values.
        """
        return tf.map_fn(tf.keras.backend.flatten, feature_values)

    def calculate_initial_thresholds(self, flattened_feature_values, quantile_values):
        """Calculates the initial thresholds for node splits based on the percentiles of flattened feature values.
        Params:
            flattened_feature_values (tf.Tensor)
            quantile_values (tf.Tensor): used to calculate the thresholds.
        Returns:
            tf.Tensor: Initial thresholds for node splits. """
        
        return tf.linalg.diag_part(tfp.stats.percentile(flattened_feature_values, quantile_values, axis=0))

    def update_soft_decision_temperature(self, feature_values):
        """ Updates the temperature for soft decisions based on the median absolute difference between
        feature values and node split thresholds.
        Params:
            feature_values: (Tensor), input feature values. """
        abs_diff = tf.math.abs(feature_values - self.node_split_thresholds)
        self.soft_decision_temperature.assign(tfp.stats.percentile(abs_diff, 50, axis=0))

    def activation_function(self, activation_type='sparsemax'): 
        """ This activation is used as part of an ablation study to assess different functions and 
        their impact on the models prediction capability when assess the AUC and FNR scores in 
        smaller imbalanced heterogeneous datasets. 
        Params:
            activation_type: str, type of activation functions, options include: 
            'sparsemax', 'entmax15', 'gumbel_softmax', and 'softmax'.
        Returns:
            Tensor, activated feature scores."""
        
        if activation_type == 'sparsemax':
            return tfa.activations.sparsemax(self.soft_feature_scores)
        elif activation_type == 'entmax15':
            return entmax15(self.soft_feature_scores)
        elif activation_type == 'gumbel_softmax':
            temperature = 0.5
            gumbel_noise = tf.random.uniform(tf.shape(self.soft_feature_scores), minval=0, maxval=1)
            gumbel_noise = -tf.math.log(-tf.math.log(gumbel_noise))  # Gumbel distribution
            logits_with_noise = (self.soft_feature_scores + gumbel_noise) / temperature
            return tf.nn.softmax(logits_with_noise)
        elif activation_type == 'softmax':
            return tf.nn.softmax(self.soft_feature_scores)
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def feature_values(self, inputs: tf.Tensor, training: bool = None, activation_type='softmax'):
        """
        This function was used in an ablation study to assess the efficacy of each function when applied to 
        the model on imbalanced heterogeneous data. The AUC and FNR metrics where assessed with each function. 
        The operation of the function, is as follows; Computes weighted feature values using the specified 
        activation function (currently softmax). The function also employs TensorFlow's `tf.einsum` for efficient 
        tensor algebra, enabling the transformation of input features into a specified dimensional space.

        The operation `tf.einsum('bi,ind->bnd')` Ref: https://www.tensorflow.org/api_docs/python/tf/einsum) multiplies 
        input features (`inputs`) with a transformation matrix (`feature_selectors`), aggregating over the input features' 
        dimension to produce transformed features. Here, 'b' represents the batch size, 'i' the number of input features, 'n' 
        the number of transformations, and 'd' the dimensionality of each transformed feature. The resulting tensor has 
        shape `[b, n, d]`, indicating that for each instance in the batch, and for each transformation, a weighted sum of 
        input features is computed, yielding a feature vector of dimension `d`.

        Params:
            inputs (tf.Tensor): Input features.
            training (bool): Specifies if the model is in training mode. Parameter is currently unused.
            activation_type (str): Specifies the type of activation function for feature selection.

        Returns:
            tf.Tensor: Computed feature values after applying the specified transformations.
        """
        feature_selectors = self.activation_function(activation_type)
        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        return feature_values

    def initialize_if_needed(self, inputs):
        """ As typically placed in DNN models, the function, conditionally initialises the model parameters 
        with the first batch of inputs if not already initialised. This is a fail safe to ensures the model 
        parameters are correctly configured to match the dimensions and characteristics of the input data, 
        facilitating 'dynamic parameter allocation' and supporting models that adapt to input sizes. As 
        emphasised by Popov, Morozov and Babenko (2019), the integration of DT into DNN, makes it crucial for 
        NODE to adjust depth and breadth of DT based on the input data characteristics. Hence NODE's flexible 
        model architecture, cannot pre-define all parameters without knowing the specific shape and size of 
        incoming data.

        Params:
            inputs: The initial data inputs used for model parameter initialization.
        """
        
        if not self.initialized:
            self.initialize(inputs)
            self.initialized = True

    def compute_feature_values(self, inputs):
        """ Computes the feature values, and provided inputs.
        Params:
            inputs (tf.Tensor): Input features.

        Returns:
            tf.Tensor: Computed feature values. """
        
        return self.feature_values(inputs)

    def calculate_threshold_logits(self, feature_values):
        """ Calculates the logits for thresholds by applying a transformation to the feature values.
        Params:
            feature_values computed from the inputs (tf.Tensor) 
        Returns:
            tf.Tensor: The logits for the thresholds, adjusted by the soft decision temperature
            (smoother decision boundary). """
        
        threshold_logits = (feature_values - self.node_split_thresholds) * tf.math.exp(-self.soft_decision_temperature)
        return tf.stack([-threshold_logits, threshold_logits], axis=-1)

    def compute_bins(self, threshold_logits):
        """ Computes binarised values from threshold logits using the sparsemoid activation function.
        Params:
            threshold_logits (tf.Tensor)

        Returns:
            tf.Tensor: Binarised values representing the presence or absence of features within specific bins.
        """
        return self.sparsemoid(threshold_logits)

    def calculate_bin_matches(self, bins):
        """ Calculates bin matches by applying a tensor algebra operation (einsum) to map bins to node-specific vectors.
        The operation, 'tf.einsum('btds,dcs->btdc')', multiplies the bins tensor with the node_onehot_vectors
        tensor, effectively determining the match between bins and nodes. The 'btds' dimensions represent
        the batch, time, depth, and sparse feature dimensions, respectively.'dcs' corresponds to the
        depth, classes, and sparse feature dimensions of the node vectors. The result 'btdc' provides a
        batch-wise, time-aware, depth-specific, and class-focused representation of the bin matches       
        (https://www.tensorflow.org/api_docs/python/tf/einsum).

        Params:
            bins (tf.Tensor): The binarised values indicating feature presence within bins.

        Returns:
            tf.Tensor: Tensor representing the match of bins to node-specific vectors.
        """
        return tf.einsum('btds,dcs->btdc', bins, self.node_onehot_vectors)

    def compute_response_weights(self, bin_matches):
        """ Computes response weights by reducing the bin matches across a specific axis.
        Params:
            bin_matches (tf.Tensor): The matched bins to node-specific vectors.
        Returns:
            tf.Tensor: The product of bin matches reduced across the specified axis, representing response weights. """
        
        return tf.math.reduce_prod(bin_matches, axis=-2)

    def compute_final_response(self, response_weights):
         """ Computes the final response by aggregating the response weights with the model's responses.
        Params:
            response_weights (tf.Tensor): The response weights computed from bin matches.
        Returns:
            tf.Tensor: The final response after aggregation via the tf.einsum function, representing the 
            model's final output. """
            
        return tf.reduce_sum(tf.einsum('bnd,ncd->bnc', response_weights, self.response), axis=1)

    def call(self, inputs: tf.Tensor, training: bool = None):
        """ Activates the NODE model by sequentially processing the input features through each component of the model. 
        Initially, inputs undergo optional preprocessing (if a feature_column is provided but is not used this code as all models 
        are pre-processed via the same methods), followed by batch normalization and dropout for regularization 
        (controlled by the 'training' bool). Inputs are then passed through each tree in the ensemble, with each tree's output 
        concatenated to the input features for the next tree, enriching the feature space. The final output is processed by a link 
        function for structured binary classification output.
        Parameters:
            inputs (tf.Tensor): Input features to the model.
            training (bool, optional): If True, the model is in training mode; otherwise, it's in inference mode. In this code 
            it is set to None to enable more flexibility with tensorflow integration, when controlling the behavior of dropout 
            and batch normalization.

        Returns:
            tf.Tensor: The final output of the model after processing through the ensemble of trees and the link function.
        """
  
        self.initialize_if_needed(inputs)
        feature_values = self.compute_feature_values(inputs) 
        self.update_soft_decision_temperature(feature_values) 
        threshold_logits = self.calculate_threshold_logits(feature_values)
        bins = self.compute_bins(threshold_logits)
        bin_matches = self.calculate_bin_matches(bins)
        response_weights = self.compute_response_weights(bin_matches)
        return self.compute_final_response(response_weights)

class NODE(tf.keras.Model):
    """The `NODE` class integrates neural networks (NNs) with Oblivious Decision Trees (ODT), enabling the model
    to operate as a powerful model for handling tabular data. This hybrid approach optimises the combination 
    of NNs along with the structured decision-making process of DTs."""
    
    @staticmethod
    def identity(x: tf.Tensor):
        """ The Identity function acts as a placeholder for feature transformation, by returning the input without any change. 
        The function is used when no explicit feature column transformation is required, allowing for direct processing 
        of input data without modification. Hence, the function serves by ensuring compatibility with the model's architecture, 
        providing a flexible interface for optional data preprocessing if required.
        Params:
            x (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Unaltered input tensor.
        """
        return x

    def __init__(self, neurons: int = 1, soft_tree_layer: int = 1, dropout_rate=0.1,
                 link: tf.function = tf.identity, soft_trees: int = 3, depth_of_tree: int = 4,
                 initialise_beta: float = 1., feature_column: Optional[tf.keras.layers.DenseFeatures] = None,
                 **kwargs):
        """Params:
               neurons (int): Specifies the number of neurons in each neural network (influencing model capacity).
               soft_tree_layer (int): The number of ('soft') tree layers in the NODE model, each layer potentially 
               increasing the model's ability to capture complex patterns.
               dropout_rate (float): Regularization parameter to prevent overfitting, applied after batch normalization.
               link (tf.function): A TensorFlow helper function applied to the final binary output. 
               soft_trees (int): The number of ('soft') trees in each layer, determining the ensemble's breadth and diversity.
               depth_of_tree (int): Controls the depth of each tree, affecting the granularity (smoother) of decision boundaries.
               initialise_beta (float): As mentioned above, the hyperparameter initialises feature thresholds, influencing how 
               features are split at each node in the trees.
               feature_column (Optional[tf.keras.layers.DenseFeatures]): Optional feature column for preprocessing input data. 
               Currently None, as a simple identity layer is used."""

        super(NODE, self).__init__()
        self.neurons = neurons
        self.soft_tree_layer = soft_tree_layer
        self.soft_trees = soft_trees
        self.depth_of_tree = depth_of_tree
        self.neurons = neurons
        self.initialise_beta = initialise_beta
        self.feature_column = feature_column
        self.dropout_rate = dropout_rate
        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(NODE.identity)
        else:
            self.feature = feature_column
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ensemble = [ODT(soft_trees=soft_trees,
                             depth_of_tree=depth_of_tree,
                             neurons=neurons,
                             initialise_beta=initialise_beta)
                         for _ in range(soft_tree_layer)]
        self.link = link

    def call(self, inputs, training=None):
        """ Calling the model to activate the NNs with in the NODE class.
        The function executes the NODE model on the input data discussed above. 
        Params:
            inputs (tf.Tensor): Input data tensor with shape [batch_size, features].
            training (bool, optional): Set to None to enable more flecibility for 
            dropout and batch normalization with tensorflow.
        Returns:
            tf.Tensor: Model output, the link function is applied with shape [batch_size, output_features]. """
        
        X = self.feature(inputs)
        X = self.bn(X, training=training)
        X = self.dropout(X, training=training)
        for i, tree in enumerate(self.ensemble):
            H = tree(X)
            X = tf.concat([X, H], axis=1)
        return self.link(H)
