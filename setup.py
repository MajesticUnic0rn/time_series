from setuptools import setup, find_packages

setup(
    name='MyPackageName', # time series forecasting
    version='1.0.0', # version 1 
    url='https://github.com/mypackage.git', # azure devops
    author='Christopher Dong', 
    author_email='',
    description='time series package and tools ',
    packages=find_packages(), 
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'], # edit for specific package for forecaster
)