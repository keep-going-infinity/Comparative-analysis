# -----------------------------------------------------------
# Dissertation Project: An Empirical Study on the Classification 
# Performance of Deep Learning vs. Gradient Boosting 
# on heterogeneous tabular data
#
# This module provides functions for running and analyzing results from NODE,
# TabNet, FFNN, and XGBoost models. It includes methods for model training, 
# performance evaluation, and interpretation of the classification results.
#
# Author: Adam Mabrouk
# Supervisor: Ben Ralph
# Institution: University of Bath
# Created on: 01/01/2024
# Version: 1.0
# License: n/a

# Library Versions Used
# ----------------------

# Python version: 3.11.5 
# numpy: 1.24.3
# pandas: 2.0.3
# sklearn (scikit-learn): 1.3.0
# optuna: 3.5.0
# tensorflow: 2.15.0
# tensorflow_addons: 0.23.0
# tensorflow_probability: 0.23.0
# seaborn: 0.12.2
# shap: 0.44.0
# matplotlib: 3.7.2

# Standard libraries for Data handling
import numpy as np
import pandas as pd
from datetime import datetime
from time import time

# Imports for visualization of results
import matplotlib.pyplot as plt
import seaborn as sns

# Imports for Machine learning libraires, tools and metrics
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score, auc, roc_curve, f1_score,
    classification_report)

from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

# SHapley Additive exPlanations (Shap), for model explainability and interpretation
import shap
from pdpbox import pdp, info_plots

# Hyperparameter optimization
import optuna


class ModelResults:
    """
    This class encapsulates the classification results of the models NODE, XGBoost, TabNet and FFNN, 
    and stores the model's performance metrics, predictions, and history. 
    
    Args:
        best_model: Trained model,
        X_test: Test features,
        y_test: Test labels,
        X_train: Training features
        history: Training history,
        study (optional): The study object, used in hyperparameter optimization.

    Attributes: (In addition)
        y_predicted_probability_labels: Predicted probabilities for each class.
        y_predicted_probability_labels_positive: Predicted probabilities for the positive class.
        y_predicted_binary_labels: Predicted class labels, based on a threshold (line 90). This was adjusted 
        for the baseline FFNN model for external baseline performance
    """
    def __init__(self, best_model, X_test, y_test, X_train, model_name, dataset_name, history_df, study=None):
        self.best_model = best_model
        self.history_df = history_df
        self.study = study
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.history_df = history_df
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.y_predicted_probability_labels = best_model.predict(self.X_test)
        self.y_predicted_probability_labels_df = pd.DataFrame(self.y_predicted_probability_labels)
        self.y_predicted_probability_labels_positive = self.y_predicted_probability_labels
        self.y_predicted_probability_positive_df = pd.DataFrame(self.y_predicted_probability_labels_positive)
        self.y_predicted_binary_labels = (self.y_predicted_probability_labels_positive > 0.5).astype(int)
        self.y_predicted_binary_labels_df = pd.DataFrame(self.y_predicted_binary_labels)
        self.calculate_metrics()
        if model_name and dataset_name:
            self.save_metrics()
            self.save_history()

    def calculate_metrics(self):
        """
        This function calculates the performance metrics for all binary classification models.
        Metrics: sensitivity, specificity, accuracy, precision, AUROC, AUPRC, F1 score, g-mean,  
        sensitivity and specificity. These metrics provide a comprehensive evaluation of the 
        model's performance on both the training and testing datasets. This is a nuanced approach
        to the typical reporting process for binary classification of tabular data. The merics are later 
        calculated using skylearn.
        """
        tn, fp, fn, tp = confusion_matrix(self.y_test,  self.y_predicted_binary_labels_df).ravel()
        sensitivity = recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = accuracy_score(self.y_test, self.y_predicted_binary_labels_df)
        precision = precision_score(self.y_test, self.y_predicted_binary_labels_df)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_predicted_probability_labels_df)
        prc_precision, prc_recall, _ = precision_recall_curve(self.y_test, self.y_predicted_probability_labels_df)
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, self.y_predicted_probability_labels_df, n_bins=10)
        auroc = roc_auc_score(self.y_test, self.y_predicted_probability_labels_df)
        auprc = average_precision_score(self.y_test, self.y_predicted_probability_labels_df)
        f1 = f1_score(self.y_test, self.y_predicted_binary_labels_df)
        g_mean = np.sqrt(sensitivity * specificity)
        
        self.metrics_df = pd.DataFrame({
            
            'Metric': ['TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 'Accuracy',
                       'Precision', 'Recall', 'F1-score', 'G-Mean', 'AUROC', 'AUPRC','FPR', 
                       'TPR', "PRC_precision", "PRC_recall","fraction_of_positives", "mean_predicted_value"],
            
            'Value': [int(tp), int(tn), int(fp), int(fn), float(sensitivity), float(specificity), float(accuracy),
                      float(precision), float(recall), float(f1), float(g_mean), float(auroc), float(auprc),
                      fpr.tolist(), tpr.tolist(), prc_precision.tolist(), prc_recall.tolist(), 
                      fraction_of_positives.tolist(), mean_predicted_value.tolist()]})
        
        self.metrics = {row['Metric']: row['Value'] for _, row in self.metrics_df.iterrows()}
        print(classification_report(self.y_test, self.y_predicted_binary_labels_df))

    def save_metrics(self):
        """
        This function saves the metrics data which is later used to calculate the mean and standard deviation 
        of all run times of the models. The functions are uniquely names with year and time but adjusted later 
        with the model name later for clarity for the reader. 
        """
        time = datetime.now() 
        new_time = time.strftime("%Y%m%d%H%M%S") 
        filename = f"../models_result_metrics/{self.model_name}/{self.model_name}_{self.dataset_name}/Saved_Metrics_{self.model_name}_{new_time}.csv"
        self.metrics_df.to_csv(filename, index=False)

    def save_history(self):
        time = datetime.now()
        new_time = time.strftime("%Y%m%d%H%M%S")
        filename = f"../log_loss_results/{self.model_name}/{self.model_name}_{self.dataset_name}/Saved_History_learning_curve_{self.model_name}_{new_time}.csv"
        self.history_df.to_csv(filename, index=False)



    def display_metrics(self):
        """
        This function displays a summary of the model's performance metrics mentioned above in calculate metrics.
        The metrics are reported with two decimal places, also with the percentage. These results are used as performance 
        indicators for the models.
        """
        metrics_df = self.metrics_df.copy()
        metrics_df = metrics_df[~metrics_df['Metric'].isin([
            "TN", "FP", "FN", "TP", "AUROC", "AUPRC", "FPR", "TPR", "PRC_precision", 
            "PRC_recall","fraction_of_positives", "mean_predicted_value"])]
        
        metrics_df['Value'] = pd.to_numeric(metrics_df['Value'], errors='coerce')
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: '{:.2f}'.format(x))
        metrics_df['Percentage'] = (metrics_df['Value'].astype(float) * 100).apply(lambda x: '{:.1f}%'.format(x))
        print(metrics_df)

    def confusion_matrix(self, filename=None):
        """
        This function displays the confusion matrix heatmap.
        Args:
            filename
        """
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_predicted_binary_labels_df).ravel()
        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives: ", tp)
        confusion_matrix_heatmap = np.array([[tn, fp],[fn, tp]])
        
        labels = np.array([["True Negative: " + str(tn), "False Positive: " + str(fp)],
                           ["False Negative: " + str(fn), "True Positive: " + str(tp)]])

        sns.heatmap(confusion_matrix_heatmap, annot=labels, fmt='', cmap='Blues',
                    xticklabels = ["Predicted 0", "Predicted 1"],
                    yticklabels = ["Actual 0", "Actual 1"])
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def auroc(self, filename=None):
        """
        This function calculates and plots the AUROC.
        Args:
            Filename
        """
        fpr = self.metrics["FPR"]
        tpr = self.metrics["TPR"]
        auroc = self.metrics["AUROC"]  

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='b', label='ROC curve (AUROC = %0.2f)' % auroc)  
        plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def auprc(self, filename=None):
        """
        This function calculates and plots the AUPRC.
        Args:
            filename
        """
        auprc = self.metrics["AUPRC"]
        precision = self.metrics["PRC_precision"]
        recall = self.metrics["PRC_recall"]

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall curve (AUC = %0.2f)' % auprc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend(loc='lower left')
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def calibration_curve(self, filename=None):
        """
        This function plots the calibration curve of the model.
        Args:
            filename
        """
        fraction_of_positives = self.metrics["fraction_of_positives"]
        mean_predicted_value = self.metrics["mean_predicted_value"]
        plt.figure()
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Model')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.legend(loc="lower right")
        plt.title('Calibration Curves')
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def model_history(self):
        """
        This function plots the model's training, validation AUC and loss over epochs.
        In the notebook, a message is printed if the history attribute is not available.
        If optuna is not used, the functions outputs the print statement below, that history 
        was not given to model results"""
        if self.history_df is not None:
            plt.plot(self.history_df['auprc'], label='Training AUPRC')
            plt.plot(self.history_df['val_auprc'], label='Validation AUPRC')
            plt.xlabel('Epochs')
            plt.ylabel('AUC')
            plt.title('AUC for Training and Validation against Epochs')
            plt.legend()
            plt.show()

            plt.plot(self.history_df['loss'], label='Training Loss')
            plt.plot(self.history_df['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training vs Validation Loss Over Epochs')
            plt.legend()
            plt.show()
        else:
            print("Cannot plot because history was not given to ModelResults")

    def xai_shap(self, filename=None):
        """
        This function generates and saves summary plots for SHapley Additive exPlanations (SHAP) 
        values to explain which feature has the most impact on model performance.
        Args:
            filename
        """
        background = self.X_train[:20].values
        explainer = shap.Explainer(self.best_model, background)
        shap_values = explainer(self.X_test[:20].values)
        shap_values.feature_names = self.X_train.columns.tolist()
        shap.summary_plot(shap_values)
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
        for i, feature in enumerate(shap_values.feature_names):
                shap.dependence_plot(i, shap_values.values,self.X_test[:20].values, shap_values.feature_names)
                if filename:
                    plt.savefig(filename.replace(".png", f"_{feature}.png"))
                    plt.close()
                else:
                    plt.show()
                    
    def optuna_trials_history(self):
        """
        This function plots the optimization history of the Optuna study if optuna
        is set to True.
        """
        if self.study:
            plt.figure(figsize=(30, 18))
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title('Optimization History', fontsize=14)
            plt.xlabel('Trials', fontsize=12)
            plt.ylabel('Objective Value', fontsize=12)
            plt.show()
        else:
            print("Cannot plot because optuna study was not given to ModelResults")
        
    def optuna_slice_plot(self):
        """
        This function generates a slice plot (using optuna's visualization tools).
        """
        if self.study:
            plt.figure(figsize=(30, 18))
            optuna.visualization.matplotlib.plot_slice(self.study)
            plt.suptitle('Slice Plot', fontsize=12)
            plt.show()
        else:
            print("Cannot plot because optuna study was not given to ModelResults")
        
    def optuna_contour_plot(self):
        """
        This function produces a contour plot (using optuna's visualization tools).
        """
        if self.study:
            plt.figure(figsize=(30, 20))
            optuna.visualization.matplotlib.plot_contour(self.study)
            plt.title('Contour Plot', fontsize=12)
            plt.show()
        else:
            print("Cannot plot because optuna study was not given to ModelResults")
        
    def optuna_parallel_plot(self):
        """
        This function produces a parallel coordinate plot (using Optuna's visualization tools).
        """
        if self.study:
            plt.figure(figsize=(20, 16))
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            plt.title('Parallel Coordinate Plot', fontsize=12)
            plt.show()
        else:
            print("Cannot plot because optuna study was not given to ModelResults")
        
    def optuna_hyperparameter_importances(self):
        """
        This function uses optuna to plot hyperparamters from each study.
        """
        if self.study:
            plt.figure(figsize=(24, 18))
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.show()
        else:
            print("Cannot plot because optuna study was not given to ModelResults")

    def measure_inference_time(self):
        """
        This function measures the inference time of the model, in combination with the 
        training and full training time for model computational complexity. If comparing classifiers, 
        the batch size must be set the same for fiar comparison. 

        Returns:
            inference_time, measured in seconds
        """
        start_time = time()
        self.y_predicted_probability_labels = self.best_model.predict(self.X_test)
        end_time = time()
        inference_time = end_time - start_time
        return inference_time
        
    def save_time_results_to_csv(self, final_training_time, inference_time):
        """
        Saves the timing results to a CSV file.

        This function records the final training time, full training time, and inference time into a CSV file. 
        This is useful for documentation and further analysis of the model's performance.

        Args:
            Final_training_time (float): Time taken for each model
            Full_training_time (float): Time taken for training, which includes hyperparameter tuning.
            Inference_time (float): Time taken for making predictions on the test set.

        Returns:
            CSV file 'time.csv' with the recorded times.
        """
        results = {
        "Final Training Time (seconds)": final_training_time,
        "Inference Time (seconds)": inference_time}
        results_df = pd.DataFrame([results])
        time = datetime.now() 
        new_time = time.strftime("%Y%m%d%H%M%S") 
        filename = f"../Time_results/{self.model_name}_results/time_{self.dataset_name}_{new_time}.csv"
        results_df.to_csv("time.csv", index=False)
        
        print("Results saved to time.csv")
        