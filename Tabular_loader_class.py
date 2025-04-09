# -----------------------------------------------------------
# Dissertation Project: An Empirical Study on the Classification 
# Performance of Deep Learning vs. Gradient Boosting 
# on heterogeneous tabular data
#
# This module provides functions for data-preprocessing before the 
# data is fed into the models NODE, TabNet, FFNN, and XGBoost. 
#
# Author: Adam Mabrouk
# Supervisor: Ben Ralph
# Institution: University of Bath
# Created on: 01/01/2024
# Version: 1.0 
# -----------------------------------------------------------

# Libraries and versions
# ----------------------
# Python version: 3.11.5 
# numpy: 1.24.3
# pandas: 2.0.3
# imbalanced-learn: 0.12.0
# scikit-learn: 1.4.0
# tensorflow: 2.15.0


# Imports for data handling and visualisation 
import numpy as np
import pandas as pd
import os

# Imports for data processing include: Tensorflow for embeddings
import tensorflow as tf

# Imports for splitting, encoding and balancing the data
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

"""
    The DataLoader is a custom made preprocessing class designed specifically to facilitate the 
    preprocessing of tabular data for deep learning (DL) tasks. The class is designed for: 
    
    1. Load data, 
    2. Drop y label,
    3. split data,
    4. Class balancing techniques (SMOTENC, Random undersampling)
    5. Subsampling: 
        a. Compressed subsample: This is one of 2 subsampling 
           stages. After the data is split and 10k extracted, the second subsample 
           takes a 5k subsample random shuffle. 
           When subsample_first=True, the specified subset size is taken, to 
           which the balancing method is then applied. This enables computational 
           efficiency for larger datasets such as lending club. 
           When subsample_first=False,one of the selected class balancing methods are 
           applied, ensuring a more representative class balance distribution. For 
           researchers with limited access to computational resource this method of 
           testing provides a viable option. 
           
    6. Remove columns,
    7. Scaling,
    8. Windsorization,
    9. Encoding choice (Label or Onehot),
   10. Embeddings: Optimisation for DL models such as TabNet.
    """
class DataLoader:
    """ 
    The class loads 4 datasets (and adaptable to others), and allows for the specification 
    of the target variable, train-test split ratio, subset size, and choice of balancing methods.

    Parameters:
    file_path: Path to CSV file datasets.
    y_label: Name of the target variable column.
    train_split_ratio: Proportion to split the data between train, validation, test.
    subset_size (optional: Size of the subset to be used for training.
    categorical_columns: List of categorical columns
    excluded_columns: user has the option for which columns to exclude. 
    use_undersampling (optional): simple technique for class balancing.
    use_oversampling (optional): Sophisticated algorithm (SMOTENC) for class balancing.
    sub_sample_first (optional): Determines the order of sub-sampling and class balancing.
    random_state: To keep reproducibility within this work, random seed is set 
    to 42. 
    """
    def __init__(self, file_path, y_label, train_split_ratio, subset_size, categorical_columns, excluded_columns, 
                 use_undersampling, use_oversampling, sub_sample_first=False, random_state=42):
        
        self.file_path = file_path
        self.random_state = random_state
        self.train_split_ratio = train_split_ratio
        self.subset_size = subset_size
        self.categorical_columns = categorical_columns
        self.excluded_columns = excluded_columns
        self.use_undersampling = use_undersampling
        self.use_oversampling = use_oversampling
        self.sub_sample_first = sub_sample_first
        self.y_label = y_label
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self):
        """
        Input: self.file_path
        Return: Data from 1 of the 4 loaded csv files.
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def drop_label(self):
        """
        Separates the target variable (y_label) from the features (self.X).
        """
        self.y = self.data[self.y_label]
        self.X = self.data.drop(columns=[self.y_label])

    def split_data(self):
        """
        Splits the dataset into train, validation, and test, based on the 
        chosen train_split_ratio (80/10/10). Stratification is used 
        to maintain the original class distribution of the target variable
        across all splits (train, val, test).
        
        Returns:
        Split datasets (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            self.X, self.y, test_size=1 - self.train_split_ratio, random_state=self.random_state, stratify=self.y)
        
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp)

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def class_balance(self):
        """
        Applies a choice of class balancing techniques (random undersampling or oversampling: SMOTENC)
        to the training data. NOTE: Class balancing is not applied to the test data. 
        """
        if self.use_oversampling:
            categorical_indices = [self.X_train.columns.get_loc(col) for col in self.categorical_columns]
            sampler = SMOTENC(random_state=self.random_state, categorical_features=categorical_indices)
            self.X_train, self.y_train= sampler.fit_resample(self.X_train, self.y_train)
        if self.use_undersampling:
            sampler = RandomUnderSampler(random_state=self.random_state)
            self.X_train, self.y_train= sampler.fit_resample(self.X_train, self.y_train)

    def sub_sample(self):
        """
        Creates a sub-sample of the training data. Once chosen, subsampling can either be 
        done prior or post class balancing depending on whether sub_sample_first is either 
        True or False.
        """
        if self.subset_size:
            perm = np.random.permutation(len(self.X_train))[:self.subset_size]
            self.X_train = self.X_train.iloc[perm]
            self.y_train = self.y_train.iloc[perm]

    def remove_columns(self):
        """Removes columns listed in excluded_columns option (below) from the dataset."""
        self.data.drop(columns=self.excluded_columns, inplace=True)

    def get_data(self):
        """
        This function loads, removes columns, separates target variable, 
        splits the data, apply's class balancing, and sub-sampling. Integral 
        to the entire preprocessing pipleline.

        Returns:
        Datasets (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        self.load_data()
        self.remove_columns()
        self.drop_label()
        self.split_data()
        
        if self.sub_sample_first:
            self.sub_sample()
            self.class_balance()
        else:
            self.class_balance()
            self.sub_sample()
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

class DataProcessor:
    """
    The DataProcessor is the second part of the custom made Data loading script, that carries out
    preprocessing of tabular data for deep learning (DL) tasks. The class is designed for flexibility 
    in preprocessing steps that are based on user requirements:
    
    Attributes:
    categorical_columns: List of categorical columns
    encoding_type (optional): Type of encoding to use, either 'label' or 'onehot'.
    use_embeddings (optional): User choice depends on the classification model.

    Methods:
    separate_numerical_features: As per naming convention, separates numerical from categorical columns
    windsorize: Applies Windsorization to reduce extreme values.
    scale_features: Scales numerical features using standard scaling.
    encode: Encodes categorical features using either 'onehot' or 'label'
    transform: Transforms the data with the mentioned preprocessing steps.
    create_embeddings: Creates embedding layers for categorical features (specific to model choice)
    apply_embeddings
    fit: The preprocessing steps are 'fit' to the training data.
    Encoding and embeddings are set to False default.
    """
    def __init__(self, categorical_columns, encoding_type=False, use_embeddings=False):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.numerical_features_columns = None
        self.limits = None
        self.categorical_columns = categorical_columns
        self.encoding_type = encoding_type
        self.use_embeddings = use_embeddings and encoding_type == 'label'
        self.embedding_layers = {}

    def separate_numerical_features(self, X_train, fit=False):
        """
        This function separates the numerical features from the dataset 
        for scaling and also converts the categorical columns to the 
        'category' data type, (excluding them from the numerical features).

        Parameters:
        X_train,
        fit

        Returns:
        Numerical features only.
        """
        for col in self.categorical_columns:
            X_train[col] = X_train[col].astype('category')
            numerical_features = X_train[[
                col for col in X_train.columns if col not in self.categorical_columns]].select_dtypes(
                include=['int64', 'float64'])
        if fit:
            self.numerical_features_columns = numerical_features.columns
        return numerical_features

    def windsorize(self, numerical_features, fit=False):
        """
        This function applies windsorization to the numerical features reducing 
        the impact of extreme outliers.

        Parameters:
        numerical_features (Data).
        fit: computes and stores the windsorization limits below and above 
        (5th-95th percentile).

        Returns:
        Windsorized numerical features.
        """
        if fit:
            self.limits = np.percentile(numerical_features, [5, 95])
        return numerical_features.clip(lower=self.limits[0], upper=self.limits[1])

    def scale_features(self, winsorized_features, fit=False):
        """
        This function scales the windsorized features using standard scaling 
        (mean 0, variance 1). 

        Parameters:
        winsorized_features,
        fit, scaler to the features,

        Returns:
        Scaled features.
        """
        if fit:
            self.scaler.fit(winsorized_features)
        return self.scaler.transform(winsorized_features)

    def encode(self, X_train):
        """
        Encodes the categorical features using either label or one-hot encoding. 
        Label encoding maps each categorical variable to a numerical value.  
        One-hot encoding creates binary columns for each category.

        Parameters:
        X_train categorical features to be encoded. 
        """
        for col in self.categorical_columns:
            if self.encoding_type == 'label':
                lbl = LabelEncoder()
                lbl.fit(X_train[col].values)
                self.label_encoders[col] = lbl
            elif self.encoding_type == 'onehot':
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                ohe.fit(X_train[[col]])
                self.onehot_encoders[col] = ohe

        if self.use_embeddings:
            self.create_embeddings(X_train)

    def transform(self, X):
        """
        This function applies windsorization, scaling, encoding, and optional embeddings. 
        The method is used to transform both training and test datasets.

        Parameters:
        X data to undergo transformation

        Returns:
        Transformed dataset.
        """
        numerical_features = self.separate_numerical_features(X, fit=False)
        windsorized_features = self.windsorize(numerical_features, fit=False)
        scaled_features = self.scale_features(windsorized_features, fit=False)
        scaled_features_df = pd.DataFrame(scaled_features, columns=self.numerical_features_columns, index=X.index)
        
        X_categorical = X[self.categorical_columns]
        X_categorical_encoded = pd.DataFrame(index=X.index)
        for col in self.categorical_columns:
            if self.encoding_type == 'label':
                X_categorical_encoded[col] = self.label_encoders[col].transform(X[col].values)
            elif self.encoding_type == 'onehot':
                encoded = self.onehot_encoders[col].transform(X_categorical[[col]])
                for i, category in enumerate(self.onehot_encoders[col].categories_[0][1:]):
                    X_categorical_encoded[f'{col}_{category}'] = encoded[:, i]
       
        # Choice of embeddings 
        if self.use_embeddings:

            X_categorical_embedded = self.apply_embeddings(X_categorical_encoded)

            return pd.concat([scaled_features_df, X_categorical_embedded], axis=1)
        else:
            return pd.concat([scaled_features_df, X_categorical_encoded], axis=1)

    def create_embeddings(self, X_train):
        """
        This function creates embedding layers for each categorical feature, of which the  
        size of each embedding is determined by the number of unique values in the feature.

        Parameters:
        X_train.

        Note:
        The embedding layers are stored and applied to the data in the transform method.
        """
        for col in self.categorical_columns:
            unique_values = X_train[col].nunique()
            embedding_size = min(np.ceil(unique_values / 2), 50)
            embedding_size = int(embedding_size)
            self.embedding_layers[col] = tf.keras.layers.Embedding(
                input_dim=unique_values,
                output_dim=embedding_size,
                input_length=1)

    def apply_embeddings(self, X):
        """
        This function applies the created embedding layers to the categorical features, which are set
        to True in this script. 

        Parameters:
        X categorical features.

        Returns:
        Embedded categorical features.
        """
        X_embedded = X.copy()
        for col in self.categorical_columns:
            X_embedded[col] = self.embedding_layers[col](X[col].values)
        return X_embedded

    def fit(self, X_train):
        """
        This function fits the DataProcessor class to the training data, enabling all preprocessing 
        (Windsorization, scaling, encoding, and embeddings).

        Parameters:
        X_train.
        """
        numerical_features = self.separate_numerical_features(X_train, fit=True)
        windsorized_features = self.windsorize(numerical_features, fit=True)
        scaled_features = self.scale_features(windsorized_features, fit=True)
        self.encode(X_train)
        if self.use_embeddings:
            self.create_embeddings(X_train)

    """ NOTE: Carryout corrleation analysis to remove columns due to positive or negative correlation"""

    

    
class DataSet:
    """
    The Dataset class initializes the parameters for preprocessing and model training.

    Parameters:
    file_path
    y_label
    train_split_ratio
    encoding_type 
    output_path: Saved processed datasets.
    categorical_columns (optional).
    excluded_columns (optional).
    subset_size (optional).
    use_undersampling (optional).
    use_oversampling (optional).
    use_embeddings (optional).
    sub_sample_first (optional).
    """
    
    """INSTRUCTIONS: 
    
    Below are 4 datasets with optional choices mentioned in the previous classes. The user can choose
    the following:
    
    1. Categorical columns for encoding
    2. Optional columns to be excluded and file path output. 
    3. Subset size: 10,000 is taken from each dataset then applied with a balacing method.
    4. The sub_sample_first=True (as per reasons mentioned above)
    5. Data splitting ratio
    6. Choice of sampling: use_undersampling, use_oversampling
    7. Choice of encoding: encoding_type='label', use_embeddings
    8. y label"""
    
    def __init__(self, file_path, y_label, train_split_ratio, encoding_type, 
                 output_path, categorical_columns=None, excluded_columns=None,
                 subset_size=None, use_undersampling=False,use_oversampling=False,  
                 use_embeddings=False, sub_sample_first=False):
        
        self.file_path = file_path
        self.y_label = y_label
        self.encoding_type = encoding_type
        self.output_path = output_path
        self.categorical_columns = categorical_columns or []
        self.excluded_columns = excluded_columns or []
        self.subset_size = subset_size
        self.use_undersampling = use_undersampling
        self.use_oversampling = use_oversampling
        self.use_embeddings = use_embeddings
        self.train_split_ratio = train_split_ratio
        self.sub_sample_first = sub_sample_first