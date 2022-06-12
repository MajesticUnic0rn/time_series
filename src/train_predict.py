#modeling
import numpy as np
import pandas as pd
import statsmodels as sm
from util import models

#import metric
from util.metric import evaluate

#import eng_feature 
from etl.eng_feature import check_df,lag_features,roll_mean_features,ewm_features,create_time_features,eq_nm_create

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
#sklearn pipeline 
from sklearn.pipeline import Pipeline


# can I run this entire script contained inside a compute cluster script? 

#json config needs 

# file location, input_features to be inserted, training_features used for training, prediction output location -csv blob, model param, horizon for forecaster use 

# param - input file/ out put file / horizon/ location

# import from data set or csv choose

# include training data w/o prediction table? forecasted IHS data? 

# get some json read for configs
#set self??

def data_check():
    return True # some sort of schema checker - on different columns - numericals vs categorical

def preprocess_data():
    return True

def main():

    file_location = 'FleetForecasting_Top100ProductSubCategory_WithIHSData_Weather_BYDay_V2.csv'
    logger.info(f'reading file from {file_location}')
    date_column ='EffectiveDate' # hard coded will change?  utils - to load data?
    input_data_raw = pd.read_csv(file_location ,sep ='|',parse_dates=[date_column])
    input_data_copy = input_data_raw.copy()
    input_data_copy.sort_values(by=[date_column], ascending=True,inplace=True)
    logger.info(f'shape of input data{input_data_copy.shape}')
    logger.info(f'data sorted on {date_column}')
    time_stamp_range = input_data_copy[date_column].min(), input_data_copy[date_column].max()
    logger.info( f'date time range for raw dataset: {time_stamp_range}') 

    # numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
    
    # categorical_features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    
    #Generate Features on lags and exponential weighted means
    
    logger.info( f'Running feature generation on eq_nm, lags and exponential weights') 
    input_data_copy.sort_values(by=['DRKey',date_column], ascending=False,inplace=True)
    input_data_copy = eq_nm_create(input_data_copy) ## creates eq name based on cat number and product description
    input_data_copy = lag_features(input_data_copy, [91, 98, 105, 112, 119, 126, 182, 364])
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [91, 98, 105, 112, 180, 270, 365]
    input_data_copy = ewm_features(input_data_copy, alphas, lags)
    exclude_columns = ['DRKey','Region','clean_time','Rental', 'QtyOwned', "ProductCategory_Desc",'ProductCategory_Nbl']
    feature_importance = [col for col in input_data_copy.columns if col not in exclude_columns ]
    input_data_eng_copy=input_data_copy.copy()
    # remove features not used anymore
    training_data=input_data_eng_copy[feature_importance]
    # prepare data for time features
    training_data.set_index(date_column,inplace=True)
    training_data.sort_index(inplace=True)

    #data split
    target = 'OnRent'
    
    categorical_features =['eq_nm','Division','RegionName']
    training_data=pd.get_dummies(training_data,columns=categorical_features) #identify cat variables to be split

    split_date = '2019-5-20'
    df_training = training_data.loc[training_data.index <= split_date]
    df_test = training_data.loc[training_data.index > split_date]

    # define features numerical and categorical
    # feature_list = {
    #     'numerical':'', # list of numerical data incoming
    #     'categorical':'' # division and regions , states???
    # }
    #cat variables into dummies

    X_train_df = create_time_features(df_training)
    X_test_df= create_time_features(df_test)
    logger.info( f'{len(X_train_df.columns)} amt of columns using to train model including Y') 
    logger.info( f'{X_train_df.columns} are my list of columns being used for training including Y') 

    # feature_columns = [ '','','',''
    # ]

    # should be in json config folder for model.config
    # feature_columns = [ 'Division',
    #                     'EffectiveDate',
    #                     'IHSRentalMarket',
    #                     'AvgTemp',
    #                     'AvgPrcp',
    #                     'Region',
    #                     'eq_nm',
    #                     'OnRent_lag_91',
    #                     'OnRent_lag_98',
    #                     'OnRent_lag_105',
    #                     'OnRent_lag_112',
    #                     'OnRent_lag_119',
    #                     'OnRent_lag_126',
    #                     'OnRent_lag_182',
    #                     'OnRent_lag_364',
    #                     'OnRent_roll_mean_365',
    #                     'OnRent_roll_mean_546',
    #                     'sales_ewm_alpha_095_lag_91',] #drop DR key tho
    #pull out 
    
    feature_columns=list(X_train_df.columns).remove('OnRent') 
    
    model_params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "seed": 42,
                    'linear_tree': False,
                    'learning_rate': .15,
                    'min_child_samples': 5,
                    'num_leaves': 31,
                    'num_iterations': 50
                    }

    X_train_df.columns 
    # Create and train model
    logger.info( f'model init and training') 

    mw = models.LightGradientBoostModelWrapper(
        feature_columns=feature_columns, model_params=model_params)

    mw.train(X_train_df, target)

    logger.info(f'model prediction') 

    results=mw.predict(X_test_df)
    
    logger.info( f'{results.head()}') 

    #results.to_csv('forecasting_results.csv')
    # pipeline_status = 'training'
    # if(pipeline_status == 'predict'):
    #     logger.info( f'prediction only')
    # else:
    #     logger.info(f'training') 
    #post processing 
    
    # input data columns checker to json
    
    # read in files # check column signature based on json 
    
    # mandatory columns input - EffectiveDate. ETC - Model signature 
    
    # print out stats about data 
    
    # feature engineering 
    
    #pre processing\

    # feature engineering stats - what was done - l
    # training lgbm 
    # run mlflow
    # run metrics
    # prediction

if __name__ == '__main__':

    #setup logging
    logger = get_custom_logger()
    # run main for e2e
    EXPERIMENT_NAME = 'On_Rent_Time_Series'
    mlflow.set_runName = 'lgbm predictions'# config loader on constants so I dont need to rewrite all this
    client = MlflowClient()
    experiments = client.list_experiments()
    if EXPERIMENT_NAME not in [e.name for e in experiments]:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name='simple_light'):
        main()
        #how to log entire artififacts including official mlflow model? 
        # does it matter if i log model correctly? or build out package for handling deployment myself?
        #mlflow.log_artifact()
    mlflow.end_run
