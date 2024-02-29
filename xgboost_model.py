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
# XGBoost version: 2.0.2

from xgboost import XGBClassifier

def create_xgboost(params, X_train):
    """ Initialises an XGBClassifier to serve as a bench marke model using specified parameters below.
    Params:
        params (dict): Parameters for the XGBClassifier.
        X_train: Training dataset.
    
    Returns:
        XGBClassifier instance configured with the given parameters.
    """
    return XGBClassifier(**params, eval_metric=["logloss", "aucpr"])

def optuna_xgboost_params(trial):
    """ Generates a dict() of XGBoost parameters optimised for a given Optuna trial.
    
    Parameters:
        trial (optuna.trial._trial.Trial): An Optuna trial object used for hyperparameter optimisation.
    Returns:
        dict: Suggested hyperparameters for the XGBoost model. """
    
    return {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'seed': trial.suggest_int('seed', 1, 1000), 
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'n_estimators':30, 
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }