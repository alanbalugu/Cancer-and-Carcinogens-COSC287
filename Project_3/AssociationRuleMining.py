#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from pprint import pprint

from CdcClustering import separate_by_year
from CdcClustering import scatterPlot
from CdcClustering import normalizeCDC_byQuestion


#separates CDC data by the type of cancer rate and returns dataframe with only those values
def separateCDCData(CDC_Data):
	print("separating data...")

	CDC_Data_AgeAR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Age")].copy()
	CDC_Data_CrudeR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Crude")].copy()
	CDC_Data_Number = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Number")].copy()

	#pprint(CDC_Data_AgeAR)

	return CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number

#adds another column for the binned value of the cancer rate (uses the z score to figure out bin label). Returns modified dataframe
def binRate(CDC_Data):
	new_CDC_data = CDC_Data

	def categorization(value):
		if value > 2.0:
			return 'very_high'
		elif value <= 2.0 and value > 1.0:
			return 'high'
		elif value <= 1.0 and value > -1.0:
			return 'medium'
		elif value <= -1.0 and value > -2.0:
			return 'low'
		else:
			return  'very low'
	
	#use a lambda function to apply the label
	new_CDC_data['datavalue_level'] = new_CDC_data['datavalue'].apply(lambda x: categorization(x))
    
	return new_CDC_data

#Determines the region for a particualr state based on the timezone. Returns the region as a string.
def categorization(value):
	if value in ['CT', 'DE', 'FL', 'GA', 'IN', 'KY', 'ME', 'MI', 'MD', 'MA', 'PA', 'OH', 'WV','VA','NC', 'SC', 'NY', 'VT', 'NH', 'RI', 'DC','NJ']:
		return 'EAST'
	elif value in ['ND', 'MN', 'WI', 'SD', 'NE', 'IA', 'IL', 'KS', 'MO', 'TN', 'OK', 'AR', 'MS', 'AL', 'TX', 'LA']:
		return 'CENT'
	elif value in ['MT', 'ID', 'WY', 'UT', 'CO', 'AZ', 'NM']:
		return 'MONT'
	elif value in ['HI', 'AK']:
		return 'ALK/HAW'
	else:
		return 'PACF'

#abbreviates the state name into the two letter string. Returns this string.
def abbreviate(value):

	us_state_abbrev = {
		'Alabama': 'AL',
		'Alaska': 'AK',
		'Arizona': 'AZ',
		'Arkansas': 'AR',
		'California': 'CA',
		'Colorado': 'CO',
		'Connecticut': 'CT',
		'Delaware': 'DE',
		'District of Columbia': 'DC',
		'Florida': 'FL',
		'Georgia': 'GA',
		'Hawaii': 'HI',
		'Idaho': 'ID',
		'Illinois': 'IL',
		'Indiana': 'IN',
		'Iowa': 'IA',
		'Kansas': 'KS',
		'Kentucky': 'KY',
		'Louisiana': 'LA',
		'Maine': 'ME',
		'Maryland': 'MD',
		'Massachusetts': 'MA',
		'Michigan': 'MI',
		'Minnesota': 'MN',
		'Mississippi': 'MS',
		'Missouri': 'MO',
		'Montana': 'MT',
		'Nebraska': 'NE',
		'Nevada': 'NV',
		'New Hampshire': 'NH',
		'New Jersey': 'NJ',
		'New Mexico': 'NM',
		'New York': 'NY',
		'North Carolina': 'NC',
		'North Dakota': 'ND',
		'Northern Mariana Islands':'MP',
		'Ohio': 'OH',
		'Oklahoma': 'OK',
		'Oregon': 'OR',
		'Palau': 'PW',
		'Pennsylvania': 'PA',
		'Puerto Rico': 'PR',
		'Rhode Island': 'RI',
		'South Carolina': 'SC',
		'South Dakota': 'SD',
		'Tennessee': 'TN',
		'Texas': 'TX',
		'Utah': 'UT',
		'Vermont': 'VT',
		'Virgin Islands': 'VI',
		'Virginia': 'VA',
		'Washington': 'WA',
		'West Virginia': 'WV',
		'Wisconsin': 'WI',
		'Wyoming': 'WY'}

	try:
		return us_state_abbrev[value]
	except:
		return "XX" #return XX if the state is not found


#separates the cdc data by the region label (timezone dependent). return the modified dataframe with the region column added
def separateByRegion(CDC_Data):

	new_CDC_data = CDC_Data

	#abbreviates the state name
	new_CDC_data['locationdesc'] = new_CDC_data['locationdesc'].apply(lambda x: abbreviate(x))

	#adds a new columns for the region
	new_CDC_data['region'] = new_CDC_data['locationdesc'].apply(lambda x: categorization(x))
        
	return new_CDC_data

#does association rule mining given the data frame and support and confidence values
def doAssociationRuleMining(CDC_Data, support_val = 0.1, confidence_val = 0.9):

	asssociation_CDC_data = CDC_Data

	CDC_records = []

	support_min = support_val
	confidence_min = confidence_val

	CDC_records = []

	#convert each record into a list of stirngs
	for i in range(0, len(asssociation_CDC_data.index)):
		CDC_records.append([str(asssociation_CDC_data.values[i,j]) for j in range(0, len(asssociation_CDC_data.columns))])

	#pprint(CDC_records)

	#run association rule mining
	association_rules = apriori(CDC_records, min_support = support_min, min_confidence = confidence_min, length_min = 2)
	association_results = list(association_rules)

	#print the resutls
	print("association rules results: \n")
	#pprint(association_results)

	print(CDC_Data.columns)

	for item in association_results:
		pprint(str(item.items) + " support: " + str(item.support))

#runs association rule mining on the CDC_API_Clean data
def main():
	print('main')

	cleaned_CDC_Data = pd.read_csv('CDC_API_Clean.csv' , sep=',', encoding='latin1', index_col = 0)
	cleaned_CDC_Data = separate_by_year(cleaned_CDC_Data, 2010)
	
	CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number = separateCDCData(cleaned_CDC_Data)

	cleaned_CDC_Data = CDC_Data_AgeAR.loc[CDC_Data_AgeAR.stratification1.str.contains("Overall")]

	region_CDC_Data = separateByRegion(cleaned_CDC_Data)

	#columns to remove from dataframe since they are the same for all rows or are redundant
	columns_to_drop = ['yearend', 'stratificationcategory1', 'stratification1', 
		'questionid', 'lowconfidencelimit', 'highconfidencelimit', 'geolocation','datavaluetype'] 

	region_CDC_Data.drop(columns_to_drop, axis = 1, inplace = True)
	region_CDC_Data.dropna(inplace = True)
	region_CDC_Data = region_CDC_Data.loc[region_CDC_Data['locationdesc'] != 'XX']

	pprint(region_CDC_Data)

	normalized_byQuestion_CDC = normalizeCDC_byQuestion(region_CDC_Data, 'question', 'datavalue')
	normalized_byQuestion_CDC.dropna(inplace = True)

	binnedRate_CDC_Data = binRate(normalized_byQuestion_CDC)

	pprint(binnedRate_CDC_Data)

	#test different support levels
	print("support ", 0.1)
	doAssociationRuleMining(binnedRate_CDC_Data, 0.1, 0.9)

	print("support ", 0.3)
	doAssociationRuleMining(binnedRate_CDC_Data, 0.3, 0.2)

	print("support ", 0.6)
	doAssociationRuleMining(binnedRate_CDC_Data, 0.6, 0.3)

if __name__ == '__main__':
	main()


