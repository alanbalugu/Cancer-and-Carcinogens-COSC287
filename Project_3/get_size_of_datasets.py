# This python file contains a function that counts the rows 
# and columns of a dataframe

import pandas as pd
from pprint import pprint

"""#########################################################
 Description: counts how many rows and columns we have in
 each of our original, uncleaned datasets; this count occurs 
 before the cleaning process, which includes the deletion of 
 certain rows containing null values
 params: none, returns: none
""" ########################################################
def count_rows_of_data():
	# file names of all the original datasets we worked with
	file_names = ["CDC_API.csv", "USCS_CancerTrends_OverTime_ByState.csv",
	"epa_data_state_chems_and_releases.csv", "combined_twitter_files.csv",
	"merged_data.csv"]

	# initialize vars to track row and column count for dataframes
	row_count = 0
	column_count = 0

	# initialize empty dataframe
	df = pd.DataFrame()

	for i in file_names:
		df = pd.read_csv(i , sep=',', encoding='latin1')
		row_count = len(df)
		column_count = len(df.columns)		
		print("For dataset " + i + ", row count is " + str(row_count) + 
			" and column count is " + str(column_count) + "\n")

def main():
	count_rows_of_data()

if __name__ == '__main__':
	main()
