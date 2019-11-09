import numpy as np
import pandas as pd
from pprint import pprint

state_dict = {
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

# Creates year and state columns in merged data frame by using all years and states in EPA data
# Source: https://stackoverflow.com/questions/43899666/pandas-how-do-i-repeat-dataframe-for-each-value-in-a-series/43899888
def createDataFrameTemplate(epa_data):
	years = pd.Series(epa_data['REPORTING_YEAR'].unique())
	state_list = pd.DataFrame(epa_data['STATE_ABBR'].unique())
	num_of_years = len(years)
	num_of_states = len(state_list)
	merged_frame = pd.DataFrame(np.repeat(state_list.values,num_of_years, axis = 0), columns = state_list.columns, index = np.tile(years,num_of_states)).rename_axis('YEAR').reset_index()
	merged_frame.rename(columns={0: "STATE_ABBR"}, inplace=True)
	merged_frame = merged_frame.sort_values(['YEAR','STATE_ABBR'])
	return merged_frame

def epaOneHotEncoding(epa_data):
	release_types_list = epa_data['CATEGORY'].unique()
	new_cols_list = []
	release_type_list = []
	for release_type in release_types_list:
		release_type_list.append(release_type)
		column_name = "AVG_REL_EST_" + release_type.replace(' ','_').upper()
		new_cols_list.append(column_name)
		epa_data[column_name] = np.where(epa_data['CATEGORY']==release_type, epa_data['AVG_REL_EST'], 0)
	return epa_data, new_cols_list, release_type_list

def insertEpaData(epa_data, new_cols_list, release_type_list, merged_frame):
	merged_frame = merged_frame.reindex( columns = merged_frame.columns.tolist() + new_cols_list) 	# add EPA columns to merged frame
	merged_frame = merged_frame.reset_index(drop=True)

	# search epa dataframe for corresponding year/state combo to populate new columns
	row_index = 0
	for state in merged_frame['STATE_ABBR']:
		year = merged_frame.loc[row_index, 'YEAR']
		for i in range(len(new_cols_list)):
			try:
				merged_frame.at[row_index,new_cols_list[i]] = epa_data[(epa_data['STATE_ABBR']==state) & (epa_data['REPORTING_YEAR']==year) & (epa_data['CATEGORY'] == release_type_list[i])] [new_cols_list[i]]
			
			# release type not found for that state/year. can be ignored
			except:	
				pass

		row_index+=1

	# add total carcinogens released as a column
	merged_frame['AVG_REL_EST_TOTAL'] = merged_frame[new_cols_list].sum(axis=1)

	return merged_frame

def insertCdcData(cdc_data, merged_frame):
	# convert state names to state codes
	cdc_data = cdc_data.replace({"Area": state_dict})

	# search cdc dataframe for corresponding year/state combo to populate merged frame
	merged_frame = merged_frame.reindex( columns = merged_frame.columns.tolist() + ['AGE_ADJUSTED_CANCER_RATE'])
	merged_frame = merged_frame.reset_index(drop=True)
	row_index = 0
	for state in merged_frame['STATE_ABBR']:
		year = merged_frame.loc[row_index, 'YEAR']
		try:
			merged_frame.at[row_index,'AGE_ADJUSTED_CANCER_RATE'] = cdc_data[(cdc_data['Area']==state) & (cdc_data['Year']==year) ] ['AgeAdjustedRate']
		
		# year/state combo not found in cdc data. can be ignored
		except:	
			pass

		row_index+=1 
	return merged_frame
	
def mergeData():

	epa_data = pd.read_csv("./EPA/cleaned/epa_data_state_releases_cleaned.csv", sep=',', encoding='latin1')
	cdc_data = pd.read_csv("./CDC/USCS_CancerTrends_OverTime_ByState.csv", sep=',', encoding='latin1')

	# adds years and states to dataframe
	merged_frame = createDataFrameTemplate(epa_data)

	print("please wait...takes ~1-2 mins to run")
	# PREP EPA DATA
	# split category column containing release types into seperate columns for each release type using one-hot encoding
	epa_data, new_cols_list, release_type_list = epaOneHotEncoding(epa_data)

	# INSERT EPA DATA INTO MERGED FRAME
	merged_frame = insertEpaData(epa_data, new_cols_list, release_type_list, merged_frame)
	
	# INSERT CDC DATA INTO MERGED FRAME
	merged_frame = insertCdcData(cdc_data, merged_frame)

	# output final merged data frame
	merged_frame.to_csv("merged_data.csv", index=False)


def main():
	mergeData()


if __name__ == '__main__':
	main()