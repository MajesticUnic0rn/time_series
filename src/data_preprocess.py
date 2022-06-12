#import packages 
import numpy as np
import pandas as pd
import statsmodels as sm
from etl.eng_feature import check_df,lag_features,roll_mean_features,ewm_features,create_time_features,eq_nm_create,clean_string
#logging package
import logging
import logging.config
import codecs
import yaml
from util.custom_log import CustomFormatter, get_custom_logger
import os
import argparse
parser = argparse.ArgumentParser(description="Examples of argparse usage", epilog="With a message at the end" )
parser.add_argument("--config", help="insert location of data process config", default=0 )
args = parser.parse_args()

#click parser or default location - config file, or default would go back to log folder 
#small function for use within data process
#need yaml reader function for config reader for modeling
#schema config checker? throw errors if something doesnt seem right
# cat varibles should be in -- assert list, assert numericals, assert 
#config - 
# dev/prod - file type loader - file_location - date column -  split data - target - cat variable - numerical 
#standardize/log/normalize?

logger = get_custom_logger()

CONFIG_PATH = "../config/"

def load_config(config_name):
    """
        Helper function to load configuration file from config file - normally data processing/modeling setup
    """
    with open(os.path.join(os.path.dirname(__file__), CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def preprocess(config,input_data_copy):
    """
        data preprocessing setup
            -calculate normals lags and ewm lags for feature creation
            -clean region name strings
            -product number and description should mash together for equipment name(eq_nm)
            -create time features from dates
            -dummify all categorical variables
            -TODO in the future - season,SIC code, bigger cat classes of eq, utilization?
    """
    date_column = config['time_stamp_column']
    logger.info(f'shape of input data{input_data_copy.shape}')
    logger.info(f'data sorted on {date_column}')
    time_stamp_range = input_data_copy[date_column].min(), input_data_copy[date_column].max()
    logger.info( f'date time range for raw dataset: {time_stamp_range}') 
    logger.info( f'Running feature generation on eq_nm, lags and exponential weights') 
    input_data_copy.sort_values(by=['DRKey',date_column], ascending=False,inplace=True)

    #lag features for on rent and clean strings
    input_data_copy = eq_nm_create(input_data_copy) ## creates eq name based on cat number and product description
    input_data_copy = lag_features(input_data_copy,config['lags'] ) #yaml config constraint 
    alphas = config['ewm_alpha'] # ymal config for parameters
    lags = config['ewm_lags'] 
    input_data_copy = ewm_features(input_data_copy, alphas, lags)
    input_data_copy['RegionName'] = clean_string(input_data_copy['RegionName'])
    # remove features not used anymore
    logger.info( f'exclude features -might not be needed pretty soon') 
    exclude_columns = ['Region','clean_time','Rental', 'QtyOwned', "ProductCategory_Desc",'ProductCategory_Nbl'] # not for me to decide to incorporate this- modelers decision
    feature_importance = [col for col in input_data_copy.columns if col not in exclude_columns ]
    input_data_eng_copy=input_data_copy.copy() # memory constraint?
    training_data=input_data_eng_copy[feature_importance]
    # prepare data for time features
    logger.info( f'create time features')
    training_data.set_index(date_column,inplace=True)
    training_data.sort_index(inplace=True)
    training_data=create_time_features(training_data)
    #add seasons based on months
    #data split
    target = config['label_column'] # yaml config
    
    logger.info( f'generate cat features') 

    categorical_features = config['categorical_feature'] #yaml config

    training_data=pd.get_dummies(training_data,columns=categorical_features)
    
    #can i set pandas categorical?

    return training_data

def main(config,is_splitable):

    file_location = config['file_location']

    logger.info(f'reading file from {file_location}')

    date_column = config['time_stamp_column']

    input_data_raw = pd.read_csv(file_location ,sep ='|',parse_dates=[date_column])

    input_data_copy = input_data_raw.copy() # memory constraint? # compare changes??? shape and column names added?

    preprocess_df = preprocess(config,input_data_copy) #bulk feature engineer and preprocess
    
    DATA_CONFIG_PATH = config['output_location']
    
    logger.info( f'data config path: {DATA_CONFIG_PATH}')
    
    if (is_splitable):
        split_date = config['split_date']
        df_training = preprocess_df.loc[preprocess_df.index <= split_date]
        df_test = preprocess_df.loc[preprocess_df.index > split_date] #split data function - we have it somewhere????
        df_test.to_csv(os.path.join(os.path.dirname(__file__), DATA_CONFIG_PATH, 'test.csv')) # create some IO function to output data into azure/sql/
        df_training.to_csv(os.path.join(os.path.dirname(__file__), DATA_CONFIG_PATH, 'train.csv'))
    else:
        preprocess_df.to_csv(os.path.dirname(__file__), DATA_CONFIG_PATH, 'predict') 

#data processing and split information - train and predict to data folder 

#and save to dataset azure

#out put data_process_out yaml output what was done - out put log into yaml? 

if __name__ == '__main__':

    logger.info( f'starting preprocessing and feature engineering')
    is_splitable = False

    if args.config:
        config_location = args.config
    else:
        config_location = "data_process_config.yml"

    config=load_config(config_location)

    if (config['preprocess']['stage'] =='dev'):
        config=config['preprocess']['stage_dev']
        is_splitable = True
    else:
        config=config['preprocess']['stage_prod']
    
    main(config,is_splitable) 