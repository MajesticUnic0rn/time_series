from setuptools import setup, find_packages

setup(
    name='onrent-timeseries', # time series forecasting
    version='1.0.0', # version 1 
    url='https://github.com/mypackage.git', # azure devops
    author='Christopher Dong', 
    author_email='99608@sunbeltrentals.com',
    description='time series package and tools for on rent prediction for fleet',
    packages=find_packages(),  # edit for specific package for forecaster
)