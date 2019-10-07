import pandas as pd
from sodapy import Socrata
from pprint import pprint
import requests
import csv

# access to public dataset doesn't require api key
# Socrata is the API provided by the CDC
source = Socrata("chronicdata.cdc.gov", None)
# using Get method to retrieve data from cdc.gov
data = source.get("u9ek-bct3", limit=15000)
# using pandas to import cdc data into dataframe
df_cdc = pd.DataFrame.from_records(data)

# writing out to file before cleaning and sorting data
def writeToFile(filename):
	df_cdc.to_csv(filename)

# cleaning data by dropping rows where no data is available
# for a certain column and deleting columns with irrelevant data
def cleanAndSortData():
	global df_cdc
	df_cdc = df_cdc[df_cdc.datavaluefootnote != 'No data available']

	#deleting unnecessary columns
	df_cdc = df_cdc.drop(['datavalueunit', 'locationabbr', 
	':@computed_region_bxsw_vy29', ':@computed_region_he4y_prf8', 
	'datavaluealt', 'datasource', 'datavaluefootnotesymbol', 
	'datavaluetypeid', 'locationid', 'stratificationcategoryid1', 
	'stratificationid1', 'topicid', 'datavaluefootnote', 'topic'], axis=1)

	# reversing order of columns for convenience of viewing
	df_cdc = df_cdc.iloc[:, ::-1]

	# sorting by year and state
	df_cdc = df_cdc.sort_values(by=['yearstart', 'locationdesc'])

# printing results to console
def print(data_frame):
	pprint(data_frame)

def main():
	# write out original dataframe (uncleaned data)
	writeToFile("CDC_API.csv")
	# clean data and delete irrelevant columns
	cleanAndSortData()
	# print out data to console
	print(df_cdc)
	# write out new clean dataframe
	writeToFile("CDC_API_Clean.csv")

if __name__ == "__main__":
	main()
