import numpy as np
import pandas as pd
from scipy.stats import zscore

del_cols = ['VAR_REL_EST','CAS_NUM','SRS_ID','LIST_3350','INACTIVE_DATE','SUM_REL_EST', 'MIN_REL_EST', 'MAX_REL_EST', 'STD_REL_EST']

STATE_CHEMS_RELEASES_FILENAME = "epa_data_state_chems_and_releases"
STATE_RELEASES_FILENAME = "epa_data_state_releases"

# START_YEAR = 
# END_YEAR = 

# then subset rows to carcinogenic and another with carcinogenic and clean air

# convert state names to state abbreviation - create a dict

def main():
	# cleanData("epa_data_state_and_releases")
	# cleanData("epa_data_state_chems_and_releases.csv")
	
	# chemicals and releases
	myDataFrame1 = readData(STATE_CHEMS_RELEASES_FILENAME)
	myDataFrame1 = dropUnnecessaryRowsColumns(myDataFrame1)
	cleanData(myDataFrame1, STATE_CHEMS_RELEASES_FILENAME)

	# releases only
	myDataFrame2 = readData(STATE_RELEASES_FILENAME)
	cleanData(myDataFrame2, STATE_RELEASES_FILENAME)

	print ("cleaning complete")

	# trimming cleaned data
	
# read data from file into pandas dataframe
def readData(file_name):
	file_name_extension = "./original/" + file_name + ".csv"
	myDataFrame = pd.read_csv(file_name_extension, sep=',', encoding='latin1')
	return myDataFrame

def dropUnnecessaryRowsColumns(myDataFrame):
	myDataFrame = myDataFrame.drop(del_cols, axis=1)
	# drop any non carcinogenic chemicals
	myDataFrame = myDataFrame[myDataFrame['CARCINOGEN']=='Y']
	return myDataFrame

def cleanData(myDataFrame, file_name):
	# drops rows with null value - without a value for every column, the row is useless
	myDataFrame.dropna(axis=0)

	# drops rows with 0 values because they aren't needed for our summation and because they take up data space
	myDataFrame['0_count'] = (myDataFrame == 0).sum(axis=1)
	myDataFrame = myDataFrame[myDataFrame['0_count'] == 0]
	myDataFrame = myDataFrame.drop('0_count',axis=1)
	myDataFrame = dropOutliers(myDataFrame)

	# output cleaned data to csv
	myDataFrame.to_csv("./cleaned/" + file_name + "_cleaned.csv", index = False)

def dropOutliers(myDataFrame):
	# zscores of AVG_REL_EST
	rel_series = myDataFrame['AVG_REL_EST']
	z_scores = (rel_series-rel_series.mean())/rel_series.std()	
	# create boolean column for whether row is an outlier or not
	outlier_list = []
	total_outliers_AVG_EST = 0
	for i in z_scores:
		if (i < -2.5) | (i > 2.5):
			total_outliers_AVG_EST += 1
			outlier_list.append(True)
		else:
			outlier_list.append(False)

	myDataFrame['OUTLIER'] = outlier_list
	myDataFrame = myDataFrame[myDataFrame['OUTLIER']==False]
	myDataFrame = myDataFrame.drop('OUTLIER',axis=1)
	print("total outliers removed: " + str(total_outliers_AVG_EST) + "\n")

	return myDataFrame


if __name__ == '__main__':
	main()