# -----------------------------------------------------------
# Dissertation Project: An Empirical Study on the Classification
# Performance of Deep Learning vs. Gradient Boosting
# on heterogeneous tabular data
#
# This python file contains the TabNet architecture.
#
# Author: Adam Mabrouk
# Supervisor: Ben Ralph
# Institution: University of Bath
# Created on: 01/01/2024
# Version: 1.0
#
# Acknowledgments:
# The tabNet architecture in this code has been adapted from the Authors:
# Arik, S.Ö. and Pfister, T., 2021, May. Tabnet: Attentive interpretable tabular learning.
# In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 8, pp. 6679-6687).
# Source paper: https://arxiv.org/pdf/1908.07442v5.pdf
# Original TabNet architecture: https://github.com/google-research/google-#research/blob/master/tabnet/tabnet_model.py
# Author: Arik, S.Ö. and Pfister, T
# Adapted TabNet architecture: Novel implementation of 'feature_dimensions' combining Nd & Na:
# This modification was adopted to achieve faster hyperparameter optimisation
# Author: https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/model.py
# -----------------------------------------------------------
# Library Versions Used
# ----------------------
# Python version: 3.11.5
# tensorflow: 2.15.0

import tensorflow as tf
from tensorflow_addons.activations import sparsemax

def create_TabNet(params, X_train):
    """This function creates the TabNet model with specified parameters
    
    Args:
        params (dict): Configuration parameters for the TabNet model, which should include keys 
        such as 'na_nd_dimensions', 'decision_steps', 'num_shared_decision_steps', 'relaxation_factor', 
        'sparsity_coefficient', 'batch_norm_momentum', and 'learning_rate'.
        X_train (array, dataframe): Training data used to define the number of features for the model.

    Returns:
        TabNet model: An instance of the TabNet model compiled with the specified parameters you see above.
    """
    tabnet = TabNet(number_of_features=X_train.shape[1], output_dim=1,
                    na_nd_dimensions=params["na_nd_dimensions"],
                    decision_steps=params["decision_steps"],
                    num_shared_decision_steps=params["num_shared_decision_steps"],
                    relaxation_factor=params["relaxation_factor"],
                    sparsity_coefficient=params["sparsity_coefficient"],
                    batch_norm_momentum=params["batch_norm_momentum"],
                    use_sparse_loss=True)

    tabnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"],
                                                      clipnorm=15),
                   
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                   metrics=[tf.keras.metrics.AUC(name='auprc', curve='PR')])
    return tabnet

def optuna_tabnet_params(trial):
    """ This function defines and returns the hyperparameter space for the TabNet model using an Optuna trial.

    Params:
        trial (optuna.trial.Trial): An Optuna trial object is used to suggest values for the parameters.
        Note: Optuna is run once (or depending on user time CPU or availability of and GPU), the trial parameters are then fed to 
        the manual params, where further runs are conducted, changing the seed each time for reproducibility.

    Returns:
        dict: A dictionary of suggested values for the TabNet model parameters.
    """
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512]),
        "na_nd_dimensions": trial.suggest_categorical("na_nd_dimensions", [32, 64, 128, 256, 512]),
        "decision_steps": trial.suggest_int("decision_steps", 2, 10, step=1),
        "num_shared_decision_steps": trial.suggest_int("num_shared_decision_steps", 0, 4, step=1),
        "seed": trial.suggest_int('seed', 1, 1000),
        "relaxation_factor": trial.suggest_float("relaxation_factor", 1., 3., step=0.1),
        "sparsity_coefficient": trial.suggest_float("sparsity_coefficient", 1e-5, 1e-1, log=True),
        "batch_norm_momentum": trial.suggest_float("batch_norm_momentum", 0.8, 1.0),
        "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        "lr_reduce_factor": trial.suggest_float("lr_reduce_factor", 0.2, 0.5),
        "lr_reduce_patience": trial.suggest_int("lr_reduce_patience", 7, 10),
        "lr_reduce_threshold": trial.suggest_float("lr_reduce_threshold", 1e-5, 1e-1, log=True)}
    
    return params

class FeatureProcessing(tf.keras.Model):
    """ The FeatureProcessing class is a component of the TabNet model responsible for processing
    input features through a sequence of operations, including a fully connected layer, batch normalisation (BN),
    and Gated Linear Unit (GLU) activation. It acts as a building block in TabNet, contributing to the model's
    decision steps by determining which features to utilise at each step. """

    def __init__(self, na_nd_dimensions, apply_glu=True, batch_norm_momentum=0.9, full_connect=None, epsilon=1e-1):
        """ Initialises the FeatureProcessing component
        
        Note: 
        Na: (attention mechanism) is the size of the output vector, used for decision making in the model.
        Nd: (decision dimension) refers to the size of the feature representation used for the prediction task.
        For simplicity and efficiency of the model, the two variables are combined

        Params:
            apply_glu (bool): Set to True, GLU activation is applied, controls the flow of information.
            
            na_nd_dimensions (int): The dimensionality of the feature space (N_a + N_d) for the fully connected 
            layer (see comments above).
            
            batch_norm (tf.keras.layers.BatchNormalization): Batch normalization layer to stabilize the training.
            batch_norm_momentum (float): Momentum for the moving average in batch normalisation.
            full_connect (tf.keras.layers.Dense, optional): An existing fully connected layer, set to None, which 
            enables a new layer to be created.
            
            epsilon (float): Kept at a small constant to avoid division by zero in batch normalisation. 
            Default set to 1e-1.

        Methods:
            call(x, training=None): Processes the input through the layers and activations in sequence order."""
        
        super(FeatureProcessing, self).__init__()
        
        self.apply_glu = apply_glu
        self.na_nd_dimensions = na_nd_dimensions
        self.full_connect = self.create_fully_connect_layer(na_nd_dimensions, apply_glu, full_connect)
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=epsilon)

    def create_fully_connect_layer(self, na_nd_dimensions, apply_glu, full_connect):
        """ Creates and returns a fully connected layer for feature processing.

        This method is responsible for either creating a new fully connected layer or returning an existing 
        one based on the provided parameters above. The layer's configuration is determined by whether the GLU 
        activation is applied and the combined dimensionality of the feature space.

        Params:
            na_nd_dimensions (int): The size of the feature representation (combined N_a and N_d dimensions).
            apply_glu (bool, optional): Set to True, GLU activation is applied, controls the flow of information.
            full_connect (tf.keras.layers.Dense, optional): An existing fully connected layer, set to None, which 
            enables a new layer to be created.

        Returns:
            tf.keras.layers.Dense: Fully connected layer, based on the specified parameter configuration.

        Note:
            If 'apply_glu' is True, the number of units in the fully connected layer is set to twice the 
            `na_nd_dimensions` to accommodate the GLU mechanism. This is based on the original implementation 
            for efficiency of the model.
            
            The weight initialise technique used in this implementation is 'glorot_uniform', (Xavier uniform initializer),
            which is the same method used in the original paper to maintain consistency (Arik et al., 2021). The 
            initialiser automatically calculates the correct scale of the initial weights of the neural network, which 
            is based on the layers configuration (input and output features) which is influenced by the 
            'na_nd_dimensions', apply_glu and the size of the previous layer."""
        
        if full_connect is not None:
            return full_connect
        units = na_nd_dimensions * 2 if apply_glu else na_nd_dimensions
        return tf.keras.layers.Dense(units, use_bias=False, kernel_initializer='glorot_uniform') 

    def glu_activation_on_input_tensor(self, x, number_of_columns=None):
        """ This function splits the input tensor into two parts: a linear and gating part. The gating part 
        undergoes sigmoid activation, then multiplication with the linear part. The mechanism allows the model 
        to focus on more relevant/pertitnent features, helping to increase model classification performance.

        Parameters:
        x (Tensor): Input tensor, typically the output from a previous layer in the model.
        number_of_columns (int, optional): The number of columns to split the input tensor into the linear
        and gating parts, if the split is not specified, it defaults to half of the columns in 'x'.

        Returns:
        Tensor 'x': The resultant tensor from GLU activation.
        """
        if number_of_columns is None:
            number_of_columns = x.shape[1] // 2

        linear_part, gating_part = tf.split(x, [number_of_columns, x.shape[1] - number_of_columns], axis=1)
        gating_signal = tf.nn.sigmoid(gating_part)
        return linear_part * gating_signal

    def call(self, x, training=None):
        """ This function executes the forward pass for the TabNet layer.

        Parameters:
        x (Tensor): The input tensor, which is processed by the layer.
        training (bool, optional): This conditional is set to None, if set to True or False, this overrides the 
        inference. The different modes determine the mean and variance for batch normalisation, leaving the default 
        False provides flexibility for the same model to be used for training and inference. 

        Returns:
        Tensor: Tensor ouput after processing through the layer.

        Raises:
        tf.errors.InvalidArgumentError: If NaN or Inf values are found in the tensor during numeric checks.
        These were added during debugging and left for the user to check for NaN or inf, as it can negatively 
        impact the training of the model.
        """
        x = self.full_connect(x)
        x = tf.debugging.check_numerics(x, "NaN or Inf in FeatureProcessing  batch normalization")
        x = self.batch_norm(x, training=training)
        x = tf.debugging.check_numerics(x, "NaN or Inf in FeatureProcessing after batch normalization")
        if self.apply_glu:
            x = self.glu_activation_on_input_tensor(x, self.na_nd_dimensions)
        return x

class FeatureTransformation(tf.keras.Model):
    """ The feature transformation class applies a sequence of transformation blocks to the input features.

    Attributes:
        num_transform_blocks (int): The total number of transformation blocks applied in the TabNet model.
        num_shared_decision_steps (int): The number of initial blocks that share decision steps. """
    def __init__(self, na_nd_dimensions, fully_connect_layers_list=None, num_transform_blocks=4, 
                 num_shared_decision_steps=2, batch_norm_momentum=0.9):
        """Params:
            na_nd_dimensions (int): The dimensions of the input features.
            fully_connect_layers_list (list): A list of fully connected layers to be used in the transformation 
            blocks.
            
            num_transform_blocks (int): The total number of transformation blocks to apply in the model.
            num_shared_decision_steps (int): The number of initial blocks that share decision steps.
            batch_norm_momentum (float): The momentum for the batch normalisation layers.
            transform_layers (list): A list of transformation blocks used to process the input features."""
        
        super(FeatureTransformation, self).__init__()
        self.num_transform_blocks = num_transform_blocks
        self.num_shared_decision_steps = num_shared_decision_steps
        self.transform_layers = self.initialise_transform_layers(na_nd_dimensions, fully_connect_layers_list, 
                                          num_transform_blocks, batch_norm_momentum)
        
    def initialise_transform_layers(self, na_nd_dimensions, fully_connect_layers_list, 
                                    num_transform_blocks, batch_norm_momentum):
        """ This function creates transformation blocks/layers for feature processing.

        params:
            na_nd_dimensions (int): The dimensions of the input features.
            fully_connect_layers_list (list): A list of fully connected layers.
            num_transform_blocks (int): The total number of transformation blocks to create.
            batch_norm_momentum (float): The momentum for the batch normalization layers.

        Returns:
            list: A list of initialised FeatureProcessing blocks. """
        transform_layers = []
        for n in range(num_transform_blocks):
            if fully_connect_layers_list and n < len(fully_connect_layers_list):
                full_connect = fully_connect_layers_list[n]
            else:
                full_connect = None
                
            transform_layers.append(FeatureProcessing(na_nd_dimensions, 
                                            batch_norm_momentum=batch_norm_momentum, 
                                            full_connect=full_connect))
        return transform_layers

    def call(self, x, training=None):
        """ This function sequentially processes the input tensor through each layer of the 
        `transform_layers` list, and after each layer (except the first), the output is scaled 
        and combined with the output of the next layer. This combination involves scaling the 
        current output by the square root of 0.5 (for normalisation) and then adding it to the 
        output of the next transformation layer (shown in the last line). Purpose: To enhance the 
        feature representation of the input tensor, with the aim of improving the model's ability 
        to learn complex patterns in tabular data.

        Params:
            x (Tensor): The input feature tensor.
            training (bool, optional): Whether the model is in inference/training mode. Defaults to None.

        Returns:
            Tensor: The transformed feature tensor, which represents enhanced feature representation 
            after processing through the layers."""
        x = self.transform_layers[0](x, training=training)
        for layer in self.transform_layers[1:]:
            x = x * tf.sqrt(0.5) + layer(x, training=training)
        return x

    @property
    def shared_fully_connect_layers_list(self):
        """ This function gets the fully connected layers of the initial shared decision steps.

        Returns:
            list: A list of fully connected layers from the initial blocks/layers."""
        return [layer.full_connect for layer in self.transform_layers[:self.num_shared_decision_steps]]

class FeatureSelectiveTransformer(tf.keras.Model):
    """ The FeatureSelectiveTransformer class is designed to selectively weigh input features using an attention 
    mechanism, which is a key attribute in the TabNet model, allowing certain features to aquire more emphasis 
    than others (feature sparsity), based on the context of the input data.

    Attributes:
        layer (FeatureProcessing): A custom layer for processing input features.

    Params:
        na_nd_dimensions (int): The dimensions of the input features.
        use_sparse_loss (bool): Flag set to True, to apply sparse loss for feature sparsity.
    """
    def __init__(self, na_nd_dimensions):
        super(FeatureSelectiveTransformer, self).__init__()
        self.layer = FeatureProcessing(na_nd_dimensions, apply_glu=False)

    def call(self, x, feature_importance_weights, training=None):
        """ This function provides a forward pass for the FeatureSelectiveTransformer.

        Params:
            x (Tensor): Input tensor.
            feature_importance_weights (Tensor): Tensor representing prior scales to apply to the attention mechanism.
            training (bool, optional): Whether the model is in inference/training mode. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the attention mechanism.
        """
        x = self.layer(x, training=training)
        return sparsemax(x * feature_importance_weights)


class TabNet(tf.keras.Model):
    """ This class, inherited from tf.keras.Model, implements the TabNet architecture for classification.
    
    Methods:
        __init__: Initializes the TabNet model with the specified parameters.
        compile_tabnet_model: Constructs the internal structure of the TabNet model.
    """
    def __init__(self, number_of_features, na_nd_dimensions, output_dim, decision_steps=2, 
                 num_transform_blocks=4, num_shared_decision_steps=2, relaxation_factor=1.5,
                 batch_norm_epsilon=1e-1, batch_norm_momentum=0.9, sparsity_coefficient=1e-1,
                 use_sparse_loss=True):  # change back to True
        """
        Attributes:
            number_of_features (int): The number of features in the input data.
            na_nd_dimensions (int): Dimensionality of the feature space, 
            and decision making process in the network.
            
            output_dim (int): Dimension of the output space.
            decision_steps (int): The number of decision steps in the network.
            num_transform_blocks (int): The number of transformation blocks used in the model.
            num_shared_decision_steps (int): The number of decision steps that are shared.
            relaxation_factor (float): Factor (gamma) used for scaling during the training process.
            sparsity_coefficient (float): Coefficient (lambda) for the sparsity regularization.
            batch_norm_epsilon (float): Epsilon value for batch normalization layers (set low).
            batch_norm_momentum (float): Momentum for the moving average in batch normalization.
            use_sparse_loss (bool, optional): Indicates whether to use sparse loss in the model. 
        Note: relaxation factor and sparsity coefficient operate in ratio. """

        super(TabNet, self).__init__()
        self.number_of_features = number_of_features
        self.na_nd_dimensions = na_nd_dimensions
        self.output_dim = output_dim
        self.decision_steps = decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_momentum = batch_norm_momentum
        self.num_transform_blocks = num_transform_blocks
        self.num_shared_decision_steps = num_shared_decision_steps
        self.use_sparse_loss = use_sparse_loss
        self.compile_tabnet_model()
        
    def initialise_batch_normalisation(self):
        """ This function creates a batch normalization layer and is used within the 
        feature transformation blocks/layers to normalise inputs for each mini-batch.

        Returns:
            tf.keras.layers.BatchNormalization: Batch normalisation layer
        """
        return tf.keras.layers.BatchNormalization(
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon)

    def construct_feature_transformation_layers(self):
        """ This function builds a sequence of feature transformation blocks/layers, one for each decision
        step, each block is responsible for transforming the input features at each decision
        step. The first block is unique, the subsequent blocks however, share fully connected layers.

        Returns:
            list[FeatureTransformation]: A list of FeatureTransformation blocks, one for each decision
            step in the TabNet model. The first block in the list is unique, and the rest share certain
            layers as per the model's architecture. """
        
        kargs = {"na_nd_dimensions": self.na_nd_dimensions + self.output_dim, 
                 "num_transform_blocks": self.num_transform_blocks, 
                 "num_shared_decision_steps": self.num_shared_decision_steps,
                 "batch_norm_momentum": self.batch_norm_momentum}

        feature_transforms = [FeatureTransformation(**kargs)]
        for i in range(self.decision_steps):
            feature_transforms.append(FeatureTransformation(
                **kargs, fully_connect_layers_list=feature_transforms[0].shared_fully_connect_layers_list))
        return feature_transforms

    def create_feature_selection(self):
        """ This method initialises a sequence of feature selection transformers, one for each 
        decision step in the model, by which each of the transformers are responsible for selecting 
        resepective relevant features at each step.

        Returns:
            list: A list of initialised FeatureSelectiveTransformers (objects)."""
        feature_selection = []
        for _ in range(self.decision_steps):
            feature_selection.append(
                FeatureSelectiveTransformer(
                    self.number_of_features))
        return feature_selection

    def init_output_layer(self):
        return tf.keras.layers.Dense(1, activation="sigmoid", use_bias=False)

    def compile_tabnet_model(self):
        """ This method initialises, configures and compiles the components for the TabNet architecture/mode, 
        such as batch normalization: feature transformation layers, feature selection layers and defines the
        final output layer. In addition this encompasses, multiple decision steps, emphasizing sequential 
        attention and decision making.

        Params:
            batch_norm (tf.keras.layers.BatchNormalization): BN layer configured with specific momentum and epsilon. 
            feature_transforms (list): List of FeatureTransformation layers for sequential feature processing.
            feature_selection (list): List of FeatureSelectiveTransformer layers for applying attention mechanisms
            to features (Hence for feature selection).
            
            head (tf.keras.layers.Dense): The output layer of the network, which has been set up with sigmoid
            activation and without bias, set=False. """
        
        self.batch_norm = self.initialise_batch_normalisation()
        self.feature_transforms = self.construct_feature_transformation_layers()
        self.feature_selection = self.create_feature_selection()
        self.head = self.init_output_layer()

    def call(self, features, training):
        
        """Params:
               features (Tensor): Input features to the model.
               training (bool, optional): Whether the model is in inference/training mode. Defaults to None.

           Returns:
              Tensor: The final output of the model after processing through all decision steps.

           Note: This method also calculates the total entropy as a measure of feature selection sparsity, 
           which is used in sparsity loss calculation. """
        
        
        num_batch_samples, output_accumulator, feature_importance_weights = self.initialize_call(features)
        attention_weighted_features = self.batch_norm(features, training=training)
        total_entropy, feature_attention_masks = 0.0, []

        for decision_step_index in range(self.decision_steps):
            x, out, mask_values = self.transform_and_activate_step(
                decision_step_index, attention_weighted_features, training)

            feature_importance_weights, attention_weighted_features, total_entropy = self.calculate_masks_and_entropy(
                decision_step_index, x, feature_importance_weights,
                total_entropy, features, training, use_modified_version=True)

            output_accumulator, attention_weighted_features = self.feature_processing_step(
                decision_step_index, output_accumulator,
                out, x, features, feature_importance_weights, training)

            entropy, attention_weights = self.feature_attention_step(
                decision_step_index, x, feature_importance_weights, training, use_modified_version=True)

            total_entropy += entropy
            feature_attention_masks.append(self.reshape_mask_for_feature_set(attention_weights))

        self.selection_masks = feature_attention_masks
        final_output = self.final_output_calculation(output_accumulator, total_entropy)
        return final_output

    def initialize_call(self, features):
        """Initializes variables for the call method.
        Params:
            features (Tensor): Input tensor.

        Returns:
            tuple: Batch size, output accumulator, feature importance weights. """
        
        num_batch_samples = tf.shape(features)[0]
        output_accumulator = tf.zeros((num_batch_samples, self.output_dim))
        feature_importance_weights = tf.ones((num_batch_samples, self.number_of_features))
        return num_batch_samples, output_accumulator, feature_importance_weights

    def feature_processing_step(self, step_index, output_accumulator, out, x,
                                features, feature_importance_weights, training,):
        """Processes features for a given decision step.
        params:
            step_index (int): Current decision step index.
            output_accumulator (Tensor): Accumulates outputs from previous steps.
            out (Tensor): Current step output.
            x (Tensor): Transformed features for current step.
            features (Tensor): Original input features.
            feature_importance_weights (Tensor): Weights for feature importance.
            training (bool): Flag that indicates training or inference mode.

        Returns:
            tuple: Updated output accumulator, attention weighted features. """
        
        if step_index > 0:
            output_accumulator += out
        attention_weights = self.feature_selection[step_index](
            x, feature_importance_weights, training=training)
        
        feature_importance_weights *= self.relaxation_factor - attention_weights
        attention_weighted_features = attention_weights * features
        return output_accumulator, attention_weighted_features

    def feature_attention_step(self, step_index, x, feature_importance_weights, training, use_modified_version):
        """Computes attention weights and entropy for a feature set.

        Params:
            step_index (int): Current decision step index.
            x (Tensor): Transformed features for current step.
            feature_importance_weights (Tensor): Weights for feature importance.
            training (bool): Flag that indicating training or inference mode.
            use_modified_version (bool): Flag for using modified calculation version.

        Returns:
            tuple: Entropy of attention weights, attention weights. """
        
        attention_weights = self.feature_selection[step_index](
            x, feature_importance_weights, training=training)
        
        entropy = self.sparse_loss(attention_weights, use_modified_version)
        return entropy, attention_weights

    def final_output_calculation(self, output_accumulator, total_entropy):
        """Calculates the final model output.
        Params:
            output_accumulator (Tensor): Accumulated outputs from all steps.
            total_entropy (float): Total entropy from all feature selections.
        Returns:
            Tensor: Final output after applying the model head/ouitput accumulator."""
        
        final_output = self.head(output_accumulator)
        self.sparsity_loss_added_to_loss_function(total_entropy)
        return final_output

    def transform_and_activate_step(self, decision_step_index, attention_weighted_features, 
                      training, activation_type='mish'):  # change back to mish
        """ This method applies feature transformation and an activation function to the input features,
        tailored for each decision step.

        Params:
            decision_step_index (int): Index of the current decision step.
            attention_weighted_features (Tensor): weighted features to be processed in the current step.
            training (bool, optional): As mentioned previously, set to default=None, indicates whether the model is 
            in training mode.
            
            activation_type (str, optional): Choice of the type of activation function to use. Defaults to 'mish'.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the transformed features, the output for the
            current step, and the mask values.

        Note: The method supports different activation functions (Mish, GLU, ReLU) and adjusts the processing
        based on the decision step index. This was implemneted during the ablation experiments """
        x = self.feature_transforms[decision_step_index](attention_weighted_features, training=training)
        mish = lambda x: x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))

        def glu(x, number_of_units=None):
            """ This function applies the Gated Linear Unit (GLU) activation to the input tensor.
            The GLU activation splits the input tensor into two halves along the last dimension.
            One half is passed through a sigmoid function (the gate), and the other half is
            the candidate activation. The final output is the element wise product of the gate
            and the candidate activation.

            Params:
                x (Tensor): The input tensor.
                number_of_units (int, optional): The number of units for the gating mechanism. If not provided, it defaults 
                to half of the last dimension of the input tensor.

            Returns:
                Tensor: The output tensor after applying the GLU activation.
            """
            if number_of_units is None:
                number_of_units = int(x.shape[-1] / 2)
            return x[:, :number_of_units] * tf.keras.activations.sigmoid(x[:, number_of_units:])

        relu = tf.keras.activations.relu

        if activation_type == 'mish' and decision_step_index > 0:
            output_activation = tf.keras.layers.Lambda(mish)(x[:, :self.output_dim])
        elif activation_type == 'glu' and decision_step_index > 0:
            output_activation = tf.keras.layers.Lambda(lambda x: glu(x, self.output_dim))(x)
        elif activation_type == 'relu':
            output_activation = tf.keras.layers.Lambda(relu)(x[:, :self.output_dim])
        else:
            output_activation = x[:, :self.output_dim]  # No activation chosen

        mask_values = x[:, self.output_dim:] if decision_step_index < self.decision_steps else None
        return x, output_activation, mask_values

    def calculate_scale_importance(self, layer_output, mask_values):
        """ This function scales the sum of the output (layer_output) of the network by the number of decision 
        steps minus one, and then multiplies it with the current mask values to update the importance of each feature.

        Params:
        layer_output (Tensor): The output tensor from the network.
        mask_values (Tensor): The current mask values for each feature.

        Returns:
        Tensor: The updated importance of each feature.
        """
        scaled_output_sum = tf.reduce_sum(
            layer_output, axis=1, keepdims=True) / (self.decision_steps - 1)
        importance_of_feature = mask_values * scaled_output_sum
        return importance_of_feature

    def calculate_masks_and_entropy(self, decision_step_index, x,
                                    feature_importance_weights, total_entropy, features, training,
                                  use_modified_version):  # change for strenght of sparsity
        
        """ This function calculates and updates mask values using the feature selection network, modifies prior scales, 
        and calculates entropy based on the current mask values. If the newly modified version is used (for ablation), it 
        also includes an additional sparsity loss.

        Params:
        decision_step_index (int): The current decision step index.
        x (Tensor): The input tensor to the feature selection network.
        mask_values (Tensor): The current mask values.
        feature_importance_weights (Tensor): The prior scales for each feature.
        total_entropy (Tensor): The cumulative entropy until the current decision step.
        features (Tensor): The original input features to the model.
        training (bool, optional): Flag, default is None=training, model remains the same either in training or inference.
        use_modified_version (bool, optional): Flag, to use the modified version with additional sparsity loss, set to False.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: Updated prior scales, attention/selected weighted features, and total entropy."""
        
        mask_values = self.feature_selection[decision_step_index](
            x, feature_importance_weights, training=training)
        feature_importance_weights *= self.relaxation_factor - mask_values
        attention_weighted_features = tf.multiply(mask_values, features)
        entropy = tf.reduce_mean(tf.reduce_sum(tf.multiply(-mask_values, tf.math.log(mask_values + 1e-1)), axis=1))

        if use_modified_version:
            additional_loss = self.sparse_loss(mask_values, use_modified_version)
            total_entropy += (entropy + additional_loss)
        else:
            # Original entropy calculation, which is left as the default (increased regularisation)
            total_entropy += entropy
        return feature_importance_weights, attention_weighted_features, total_entropy

    def reshape_mask_for_feature_set(self, mask_values):
        """ This function expands the dimensions of the mask values (tensor), 
        making it suitable for element-wise multiplication with the feature set.

        Params:
            mask_values (Tensor): The mask values tensor.

        Returns:
            Tensor: The reshaped mask values tensor. """
        return tf.expand_dims(tf.expand_dims(mask_values, 0), 3)

    def sparsity_loss_added_to_loss_function(self, total_entropy):
        """ This function calculates the sparsity loss based on the total entropy and the predefined sparsity 
        coefficient, then, adds the sparsity loss to the model's overall loss function.

        Params:
        total_entropy (Tensor): The total entropy across all decision steps."""
        self.add_loss(self.sparsity_coefficient * total_entropy / (self.decision_steps - 1))

    def sparse_loss(self, mask_values, use_modified_version):
        """ This function calculates the sparse loss which is a crucial component of the TabNet architecture.
        Spare loss encourages the model to use fewer features, thus making the model's decisions more interpretable.
        The loss is calculated by using the entropy of the mask values, with an added option to modify the strength
        of sparsity, an optimisation, hard coded, from the origical version of the loss and used in an ablation
        study. The modified version applies an additional scaling factor to the original loss to adjust the
        strength of sparsity.

        Parameters:
            mask_values (tensor): A tensor representing the mask values used in the TabNet model.
            use_modified_version (bool, optional): A flag to determine whether to use the modified version of
            the sparse loss (0.6 was found to improve AUROC score in ablation experiment).

        Returns:
            float: The calculated sparse loss, Returns 0.0 if sparse loss is not used in the model.
        Note: While mask_values are available, shap is used in this study to compare feature importance"""

        if self.use_sparse_loss:
            epsilon = 1e-1
            original_loss = tf.reduce_mean(tf.reduce_sum(-mask_values * tf.math.log(mask_values + epsilon), axis=1))

            if use_modified_version:
                sparsity_strength = 0.6  # Decrease reg < increases regularisation.
                modified_loss = sparsity_strength * original_loss
                return modified_loss
            else:
                return original_loss
        else:
            return 0.0