# -----------------------------------------------------------
# Dissertation Project: An Empirical Study on the Classification 
# Performance of Deep Learning vs. Gradient Boosting 
# on heterogeneous tabular data
#
# This script is used to train and test the classification models
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
# NumPy: 1.24.3
# Scikit-learn: 1.4.0
# Optuna: 3.5.0

# Standard libraries, for time, random seed, and operating systems
import os
from time import time
import random

# Data handling and processing
import numpy as np
import pandas as pd

# Machine Learning, preprocessing and scoring metric libraries 
from sklearn.model_selection import StratifiedKFold

# Hyperparameter optimisation
import optuna

# TensorFlow and call backs for model checkpoints, learning rate decay 
# and early stopping, to stop overfitting. #EarlyStopping
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class ModelTraining:

    def __init__(self,
                 seed, use_optuna, 
                 checkpoint_name,
                 log_directory, 
                 create_model_function, 
                 epochs, 
                 verbose,
                 n_trials, 
                 params, model_type="DL"):
        
        """ModelTraining class
        Params:
            seed: Random seed to run the model
            use_optuna: Choice of using optuna or manually selecting best params 
            checkpoint_name: Model check point record
            log_directory: record logging
            create_model_function: Relative to either DL/ML model
            epochs: Training epochs
            verbose: verbosity mode
            num_cross_val_splits: The number of cross validation splits
            n_trials: Number of trials for Optuna optimization
            params: Dict for optuna parameters 
            
        Returns:
            None"""
        
        self.seed = seed
        self.use_optuna = use_optuna
        self.checkpoint_name = checkpoint_name
        self.log_directory = log_directory
        self.create_model_function = create_model_function
        self.epochs = epochs
        self.verbose = verbose
        self.n_trials = n_trials
        self.params = params
        self.model_type = model_type
        self.sampler = self.set_seed(seed)


        self.study = optuna.create_study(direction='maximize',
                                    sampler=self.sampler) if use_optuna else None 

    def set_seed(self, seed_value):
        """This function is to ensure reproducibility of the results
        Params:
            Seed_value
        Returns:
            optuna.samplers.TPESampler: Specified seed"""
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        if seed_value >= 0:
            return optuna.samplers.TPESampler(seed=seed_value)  

    def train_model(self, params, X_train, y_train, X_val, y_val):
         """
        Trains each of the models using their respective parameters and data, and evaluates its performance.

        Params:
            params (dict): Configuration parameters for each model including hyperparameters.
            X_train (ndarray): Training data features.
            y_train (ndarray): Training data labels.
            X_val (ndarray): Validation data features.
            y_val (ndarray): Validation data labels.

        Returns:
            tuple: A tuple containing the trained model, a dataframe with training history, and training duration.
        """
        self.set_seed(self.seed)
        model = self.create_model_function(params, X_train)

        start_training = time()

        if self.model_type == "DL":
            checkpoint = ModelCheckpoint(self.checkpoint_name, monitor='val_auprc', verbose=self.verbose,
                                        save_best_only=True, save_weights_only=True, mode='max')

            learning_rate_decay = ReduceLROnPlateau(monitor='val_auprc',
                                                    factor=params['lr_reduce_factor'],
                                                    patience=params['lr_reduce_patience'],
                                                    verbose=self.verbose,
                                                    threshold=params['lr_reduce_threshold'])

            early_stop = EarlyStopping(monitor='val_auprc', patience=6, mode='max',
                                        restore_best_weights=True, verbose=self.verbose)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_directory)
            history = model.fit(X_train,
                                y_train,
                                epochs=self.epochs,
                                batch_size=params['batch_size'],
                                validation_data=(X_val, y_val),
                                callbacks=[learning_rate_decay, tensorboard_callback, early_stop],
                                verbose=self.verbose)
            history_df = pd.DataFrame(
                {col: history.history[col] for col in ["loss", "val_loss", "auprc", "val_auprc"]})
        elif self.model_type == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 
                      early_stopping_rounds=6,
                           verbose=self.verbose)
            history = model.evals_result()
            history_df = pd.DataFrame(
                {"loss": history["validation_0"]["logloss"], "val_loss": history["validation_1"]["logloss"],
                 "auprc": history["validation_0"]["aucpr"], "val_auprc": history["validation_1"]["aucpr"]})
        else:
            raise ValueError("model type is invalid / not supported")

        end_training = time()

        return model, history_df, end_training - start_training

    def optimize_parameters(self, X_train, y_train, X_val, y_val):
        """ Optimizes model parameters using a specified objective function over n trials.

        Params:
            X_train (ndarray): Training data features.
            y_train (ndarray): Training data labels.
            X_val (ndarray): Validation data features.
            y_val (ndarray): Validation data labels.

        Returns:
            dict: Best parameters found through optimisation, either manual or optuna. """
        
        def objective(trial):
            """ Objective function for hyperparameter optimisation, which evaluates model performance.
            Params:
                trial (Trial): Each trial suggests hyperparameters to evaluate.
            Returns:
                float: The evaluation metric of the model on the validation dataset, specifically the last value of 'val_auprc'. """
            
            _, history_df, _ = self.train_model(self.params(trial), X_train, y_train, X_val, y_val)
            return history_df["val_auprc"].iloc[-1]

        start_optimization = time()
        self.set_seed(self.seed)
        self.study.optimize(objective, n_trials=self.n_trials)
        best_params = self.study.best_trial.params
        print("Best params:", best_params)
        end_optimization = time()
        optimization_time = end_optimization - start_optimization
        print(f"Hyperparameter Optimization Time: {optimization_time} seconds")
        return best_params


    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """ Trains and evaluates the model(s) using the best parameters found manually or
            through Optuna.
        Params:
            X_train 
            y_train 
            X_val 
            y_val
        Returns:
            Results for trained model, training history, and final training time."""
        if self.use_optuna:
            best_params = self.optimize_parameters(
                X_train, y_train, X_val, y_val)
        else:
            best_params = self.params
        print("Best params:", best_params)

        self.set_seed(best_params["seed"])
        model, history_df, final_training_time = self.train_model(best_params, X_train, y_train, X_val, y_val)

        print(f"Final Model Training Time: {final_training_time} seconds")
        return model, history_df, final_training_time