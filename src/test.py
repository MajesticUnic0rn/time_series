import numpy as np
import pandas as pd
import statsmodels as sm
from etl.eng_feature import check_df,lag_features,roll_mean_features,ewm_features,create_time_features,eq_nm_create
#logging package
import logging
import logging.config
import codecs
import yaml
from util.custom_log import CustomFormatter, get_custom_logger
import os 
#mlflow packages for tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
#arg parser
import argparse
parser = argparse.ArgumentParser(description="Examples of argparse usage", epilog="With a message at the end" )
parser.add_argument("--config", help="insert location of data process config", default=0 )
args = parser.parse_args()
#logger
logger = get_custom_logger()

CONFIG_PATH = "../config/"

def load_config(config_name):
    with open(os.path.join(os.path.dirname(__file__), CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

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
    with mlflow.start_run(run_name=config['run_tag']):
        logger.info('ran sucessfully')
        #how to log entire artififacts including official mlflow model? 
        # does it matter if i log model correctly? or build out package for handling deployment myself?
        #mlflow.log_artifact()
    mlflow.end_run

## run this program as `python test.py --config custom_model_config.yml`