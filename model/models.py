"""
models.py
This file contains the abstraction for a model, which should include:
- model binary
- pointer to training set(s)
- metrics (or pointer to metrics)
"""
from abc import ABC, abstractmethod
#from .helpers import assert_subset
#from .io import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import lightgbm as lgbm
import shap
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import typing

RANDOM_STATE = 42
ALL_PROCESSORS = -1
IMPUTATION_VALUE = -1.0

class ModelWrapper(ABC):
    """Abstract wrapper class for a machine learning model."""

    def __init__(self, name: str, feature_columns: typing.List[str] = [], model_params: dict = {}, data_dict: dict = {}, metric_dict: dict = {}):
        """Constructor. data_dict and metric_dict store dictionaries of data paths and metric values respectively."""
        self.name = name
        self.feature_columns = feature_columns
        self.model_params = model_params
        self.data_dict = data_dict
        self.metric_dict = metric_dict
        self.model = None

    def add_feature_columns(self, feature_columns: typing.List[str]):
        """Adds a list of feature columns to the current list. No deduping."""
        self.feature_columns += feature_columns

    def add_data_paths(self, path_dict: dict):
        """Adds a dictionary of data paths."""
        self.data_dict.update(path_dict)

    def add_metrics(self, metric_dict: dict):
        """Adds a metric dictionary."""
        self.metric_dict.update(metric_dict)

    def add_data_path(self, path_key: str, path_name: str):
        """Adds a data path. Must specify key and name, i.e. key='train_df' and name='path/to/trainset.pq'"""
        self.add_data_paths({path_key: path_name})

    def add_metric(self, metric_name: str, metric_val: typing.Any):
        """Adds a metric value. Must specify name and value of the metric."""
        self.add_metrics({metric_name: metric_val})

    def get_data_paths(self) -> dict:
        """Returns the data path dictionary."""
        return self.data_dict

    def get_metrics(self) -> dict:
        """Returns the metric dictionary."""
        return self.metric_dict

    def save(self, component: str, dev: bool = True, overwrite: bool = False, version: str = None):
        """Saves a model wrapper object to s3."""
        # Do not allow a save without having at least a data path and a metric
        assert self.data_dict, 'No data paths were added.'
        assert self.metric_dict, 'No metrics were added.'

        # Call io function
        return save_output_pkl(self, component, dev, overwrite, version)

    @classmethod
    def load(cls, component: str, dev: bool = True, version: str = None):
        """Loads a model wrapper object from s3."""
        return load_output_pkl(component, dev, version) #loading output is related to s3 - possible return true to avoid errors

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def score(self):
        pass

class LightGradientBoostModelWrapper(ModelWrapper):
    def __init__(self, feature_columns: typing.List[str] = [], model_params: dict = {}):
        """Defaults to full parallelism and random state = 42."""
        base_params = {'n_jobs': ALL_PROCESSORS, 'random_state': RANDOM_STATE}
        base_params.update(model_params)
        super(LightGradientBoostModelWrapper, self).__init__(
            name='lgbm_classifier_no_preprocessing', feature_columns=feature_columns, model_params=base_params)

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Identity map for preprocessing but fill null values with -1"""
        #assert_subset(self.feature_columns, df.columns)
        return df[self.feature_columns].fillna(IMPUTATION_VALUE).values

    def train(self, df: pd.DataFrame, label_column: str):
        """Fits a random forest classifier to the data."""
        assert label_column not in self.feature_columns, 'Label column is in the feature list.'
        assert label_column in df.columns, 'Label column is not in the dataframe.'

        X = self.preprocess(df)
        y = df[label_column].values

        model = lgbm(**self.model_params)
        model.fit(X, y)
        self.model = model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns probability of the prediction being of class 1."""
        assert self.model is not None, 'Model is not trained. Please call .train(...).'
        X = self.preprocess(df)
        return self.model.predict(X)#[:, 1] uncomment if something goes wrong #the hell is predict_proba?

    def score(self, df: pd.DataFrame, label_column: str) -> float:
        """Returns F1 score (measure of precision and recall)."""
        assert label_column not in self.feature_columns, 'Label column is in the feature list.'
        assert label_column in df.columns, 'Label column is not in the dataframe.'

        rounded_preds = self.predict(df).round()
        return f1_score(df[label_column].values, rounded_preds) # replace f1 with metric evaluation

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns a dataframe corresponding to the importance for each feature."""
        assert self.model is not None, 'Model is not trained. Please call .train(...).'
        feature_importance_df = pd.DataFrame({'feature': self.feature_columns,
                                              'importance': self.model.feature_importances_})
        return feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    def get_shap_value(self ,df: pd.DataFrame)-> pd.DataFrame:
        """Returns a dataframe corresponding to the shape value for each feature."""
        assert self.model is not None, 'Model is not trained. Please call .train(...).'
        shap_values = shap.TreeExplainer(self.model).shap_values(df)
        shap.summary_plot(shap_values, df,show=False)
        plt.savefig('./images/shap.png',bbox_inches='tight')
        return shap_values