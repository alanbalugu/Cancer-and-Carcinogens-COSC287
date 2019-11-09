import numpy as np
import pandas as pd

del_cols = ['VAR_REL_EST','CAS_NUM','SRS_ID','LIST_3350','INACTIVE_DATE','AVG_REL_EST', 'MIN_REL_EST', 'MAX_REL_EST', 'STD_REL_EST']

states = {
        'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American Samoa', 'AZ': 'Arizona',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia',
        'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'GU': 'Guam', 'HI': 'Hawaii', 'IA': 'Iowa',
        'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky',
        'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan',
        'MN': 'Minnesota', 'MO': 'Missouri', 'MP': 'Northern Mariana Islands', 'MS': 'Mississippi',
        'MT': 'Montana', 'NA': 'National', 'NC': 'North Carolina', 'ND': 'North Dakota',
        'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada',
        'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
        'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
        'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin Islands',
        'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
}

states_inverted = {
        'Alaska': 'AK', 'Alabama': 'AL', 'Arkansas': 'AR', 'American Samoa': 'AS', 'Arizona': 'AZ',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'District of Columbia': 'DC',
        'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Guam': 'GU', 'Hawaii': 'HI', 'Iowa': 'IA',
        'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Kansas': 'KS', 'Kentucky': 'KY',
        'Louisiana': 'LA', 'Massachusetts': 'MA', 'Maryland': 'MD', 'Maine': 'ME', 'Michigan': 'MI',
        'Minnesota': 'MN', 'Missouri': 'MO', 'i': 'MP', 'Mississippi': 'MS',
        'Montana': 'MT', 'National': 'NA', 'North Carolina': 'NC', 'North Dakota': 'ND',
        'Nebraska': 'NE', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'Nevada': 'NV',
        'New York': 'NY', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'Puerto Rico': 'PR', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Virginia': 'VA', 'Virgin Islands': 'VI',
        'Vermont': 'VT', 'Washington': 'WA', 'Wisconsin': 'WI', 'West Virginia': 'WV', 'Wyoming': 'WY'
}


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

	# investigate min active date and max inactive data and then delete
	print("max active date = ", max(myDataFrame['ACTIVE_DATE']))

	return myDataFrame

def cleanData(myDataFrame, file_name):
	# drops rows with null value - without a value for every column, the row is useless
	myDataFrame.dropna(axis=0)

	# drops rows with 0 values because they aren't needed for our summation and because they take up data space
	myDataFrame['0_count'] = (myDataFrame == 0).sum(axis=1)
	myDataFrame = myDataFrame[myDataFrame['0_count'] == 0]

	# output cleaned data to csv
	myDataFrame.to_csv("./cleaned/" + file_name + "_cleaned.csv", index = False)


if __name__ == '__main__':
	main()