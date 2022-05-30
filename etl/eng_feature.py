# helper function - input this into python for reusability
import pprint 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Lag/Shifted Features
def lag_features(df, lags):
    for lag in lags:
        df['OnRent_lag_' + str(lag)] = df.groupby(["Division", "ProductCategory_Nbl"])['OnRent'].transform(
            lambda x: x.shift(lag))
    return df

# Rolling Mean Features
def roll_mean_features(df, windows):
    for window in windows:
        df['OnRent_roll_mean_' + str(window)] = df.groupby(["Division", "ProductCategory_Nbl"])['OnRent']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean())
    return df

# Exponentially Weighted Mean Features
def ewm_features(df, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            df['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                df.groupby(["Division", "ProductCategory_Nbl"])['OnRent'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return df
    
def create_time_features(df, target=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(df['dayofyear'])
    df['cos_day'] = np.cos(df['dayofyear'])
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    X = df.drop(['date'], axis=1)
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y
    return X


def eq_nm_create(df):
    cols = ['ProductCategory_Nbl', 'ProductCategory_Desc']
    df['ProductCategory_Desc'] = df['ProductCategory_Desc'].str.replace(r"[\"\',< ]", '')
    df['eq_nm'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return df

def check_df(df, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Tail #####################")
    print(df.tail(tail))
    print("##################### NA #####################")
    print(df.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
