# Performance of Deep Learning vs. Gradient Boosting
# on heterogeneous tabular data
#
# This python file contains the FFNN architecture
#
# Author: Adam Mabrouk
# Supervisor: Ben Ralph
# Institution: University of Bath
# Created on: 01/01/2024
# Version: 1.0
# -----------------------------------------------------------
# Library Versions Used
# ----------------------
# Python version: 3.11.5
# tensorflow: 2.15.0
# Keras version: 2.15.0

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

def create_feed_forward_neural_network(params, X_train):
    """ The FFNN architecture includes two hidden layers with ReLU activation and dropout for regularisation, 
    followed by a sigmoid output layer for binary classification. The model is compiled with the Adam optimizer,
    binary crossentropy loss, and tracks AUPRC training and validation.
    
    Params:
        params, dict containing hyperparameters for the model. 
        "units_1" (int): Number of neurons in the first hidden layer.
        "dropout_1" (float): Dropout rate for the first hidden layer.
        "units_2" (int): Number of neurons in the second hidden layer.
        "dropout_2" (float): Dropout rate for the second hidden layer.
        "learning_rate" (float): Learning rate for the Adam optimizer.
        X_train (numpy.ndarray): Training data features.
    
    Returns:
        model (tensorflow.keras.Model): Keras model ready for training with binary classification."""
    
    model = Sequential()
    model.add(Dense(units=params["units_1"], input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(params["dropout_1"]))
    model.add(Dense(units=params["units_2"], activation='relu'))
    model.add(Dropout(params["dropout_2"]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=params["learning_rate"]),
                  metrics=['accuracy', AUC(name='auprc', curve='PR')])

    return model

def optuna_ffnn_params(trial):
    """ This function defines and returns the hyperparameter space for the FFNN model using an Optuna trial.

    Params:
        trial (optuna.trial.Trial): Optuna trial object is used to suggest values for the parameters.
        Note: Optuna is run once (or depending on user time), the trial parameters are then fed to 
        the manual params, where further runs are conducted, where a dual seed is applied each time for reproducibility.

    Returns:
        dict of suggested values for the FFNN model parameters.
    """
    return {
        "dropout_rate": trial.suggest_float('dropout_rate', 0.1, 0.5),
        'units_1': trial.suggest_int('units_1', 32, 512),
        'dropout_1': trial.suggest_float('dropout_1', 0.0, 0.5), #trial.suggest_float('dropout_1', 0.0, 0.5)
        'units_2': trial.suggest_int('units_2', 32, 512),
        'dropout_2': trial.suggest_float('dropout_2', 0.0, 0.5),
        'seed': trial.suggest_int('seed', 1, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        "lr_reduce_factor": trial.suggest_float("lr_reduce_factor", 0.2, 0.5),
        "lr_reduce_patience": trial.suggest_int("lr_reduce_patience", 9, 10),
        "lr_reduce_threshold": trial.suggest_float("lr_reduce_threshold", 0.0005, 0.001, log=False)}
