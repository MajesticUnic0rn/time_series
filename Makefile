
#make file for reinstall onrent time series for packaging
# do some more research on make files to clean it up
reinstall:
	pip uninstall onrent-timeseries 
	pip install ../OnRent_Time_Series
