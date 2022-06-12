#modeling
import numpy as np
import pandas as pd
import statsmodels as sm
from util import models
#import metric
from util.metric import evaluate
#visual
import shap
import seaborn as sns
from matplotlib import pyplot as plt
#mlflow packages
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
##logging packages
import logging
import logging.config
import codecs
import yaml
from util.custom_log import CustomFormatter, get_custom_logger
#arg parse for configuration
import argparse
import os

parser = argparse.ArgumentParser(description="Examples of argparse usage", epilog="With a message at the end" )
parser.add_argument("--config", help="insert location of data process config", default=0 )
args = parser.parse_args()
#logger
logger = get_custom_logger()

CONFIG_PATH = "../config/"

def load_config(config_name):
    """
    Helper function to load configuration file from config file - normally data processing/modeling setup
    """
    with open(os.path.join(os.path.dirname(__file__), CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def hyperparam_tuning():
    return True


def main(train_df, test_df, config):
    """
    Utilize custom model wrapper - input training and testing data set. Config file should have model params for model running
    Returns: Custom Model for further prediction and evaluation
    """
    mw = models.LightGradientBoostModelWrapper(
            feature_columns=config['feature'],
            model_params=config['model_param'])
    mw.train(train_df, config['label_column'])
    logger.info(f'model prediction')
    return mw

if __name__ == '__main__':
    #setup logging
    if args.config:
        config_location = args.config
    else:
        config_location = "model_config.yml"

    config=load_config(config_location)

    # run main for e2e
    EXPERIMENT_NAME = config["experiment_name"]
    client = MlflowClient()
    experiments = client.list_experiments()
    
    if EXPERIMENT_NAME not in [e.name for e in experiments]:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # for each region train model and predict

    
    with mlflow.start_run(run_name=config['run_tag']):
        model_config=config['lgbm_model']
        logger.info('reading training and test data')
        
        train_df=pd.read_csv(os.path.join(os.path.dirname(__file__), model_config['data']['train']))
        test_df=pd.read_csv(os.path.join(os.path.dirname(__file__), model_config['data']['test']))
        
        logger.info('Training time series model')
        model=main(train_df, test_df, model_config)
        logger.info(f'model_metrics')
        model_metrics=model.score(test_df, model_config['label_column'])
        mlflow.log_metrics(model_metrics)
        # run predictions
        prediction = model.predict(test_df)
        test_df['prediction']=prediction
        test_df.to_csv((os.path.join(os.path.dirname(__file__), model_config['data']['predict'])))
        logger.info(f'running feature importance')
        feature_imp=model.get_feature_importances()
        plt.figure(figsize=(20, 10))
        logger.info(f'{feature_imp.head()}')
        data=feature_imp.sort_values(by="importance", ascending=False).head(30)
        sns.barplot(x="importance", y="feature", data=data)
        plt.title('LightGBM Features (avg over folds)')
        plt.savefig('../images/lgbm_importances-01.png')
        plt.clf()
        mlflow.log_artifacts("../images", artifact_path="diagnostics")
        mlflow.log_artifacts("../config", artifact_path="config")
        #how to log entire artifacts including official mlflow model? 
        # does it matter if i log model correctly? or build out package for handling deployment myself?
        #mlflow.log_artifact()
    mlflow.end_run