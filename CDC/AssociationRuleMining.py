import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from pprint import pprint

from CdcClustering import separate_by_year
from CdcClustering import scatterPlot
from CdcClustering import normalizeCDC_byQuestion

def separateCDCData(CDC_Data):
	print("separating data...")

	CDC_Data_AgeAR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Age")].copy()
	CDC_Data_CrudeR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Crude")].copy()
	CDC_Data_Number = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Number")].copy()

	#pprint(CDC_Data_AgeAR)

	return CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number


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
	
	new_CDC_data['datavalue_level'] = new_CDC_data['datavalue'].apply(lambda x: categorization(x))
        

	return new_CDC_data

def categorization(value):
	if value in ['CT', 'DE', 'FL', 'GA', 'IN', 'KY', 'ME', 'MI', 'MD', 'MA', 'PA', 'OH', 'WV','VA','NC', 'SC', 'NY', 'VT', 'NH', 'RI', 'DC','NJ']:
		return 'EAST'
	elif value in ['ND', 'MN', 'WI', 'SD', 'NE', 'IA', 'IL', 'KS', 'MO', 'TN', 'OK', 'AR', 'MS', 'AL', 'TX', 'LA']:
		return 'CENT'
	elif value in ['MT', 'ID', 'WY', 'UT', 'CO', 'AZ', 'NM']:
		return 'MONT'
	else:
		return 'PACF'

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
		return "XX"

def separateByRegion(CDC_Data):

	new_CDC_data = CDC_Data


	new_CDC_data['locationdesc'] = new_CDC_data['locationdesc'].apply(lambda x: abbreviate(x))


	new_CDC_data['region'] = new_CDC_data['locationdesc'].apply(lambda x: categorization(x))
        
	return new_CDC_data

def main():
	print('main')

	cleaned_CDC_Data = pd.read_csv('CDC_API_Clean.csv' , sep=',', encoding='latin1', index_col = 0)
	cleaned_CDC_Data = separate_by_year(cleaned_CDC_Data, 2010)
	
	CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number = separateCDCData(cleaned_CDC_Data)

	cleaned_CDC_Data = CDC_Data_AgeAR.loc[CDC_Data_AgeAR.stratification1.str.contains("Overall")]
	#cleaned_CDC_Data = CDC_Data_AgeAR.loc[CDC_Data_AgeAR.question.str.contains("mortality")]

	region_CDC_Data = separateByRegion(cleaned_CDC_Data)

	#keep question, year start, data value, locationdesc

	columns_to_drop = ['yearend', 'stratificationcategory1', 'stratification1', 
		'questionid', 'lowconfidencelimit', 'highconfidencelimit', 'geolocation','datavaluetype'] 

	region_CDC_Data.drop(columns_to_drop, axis = 1, inplace = True)
	region_CDC_Data.dropna(inplace = True)
	region_CDC_Data = region_CDC_Data.loc[region_CDC_Data['locationdesc'] != 'XX']


	#cleaned_CDC_Data.dropna(inplace = True)
	#ENCODE DATA QUALITATIVELY by BINNING?

	pprint(region_CDC_Data)

	normalized_byQuestion_CDC = normalizeCDC_byQuestion(region_CDC_Data, 'question', 'datavalue')
	normalized_byQuestion_CDC.dropna(inplace = True)

	binnedRate_CDC_Data = binRate(normalized_byQuestion_CDC)

	pprint(binnedRate_CDC_Data)

	CDC_records = []

	support_min = 0.1
	confidence_min = 0.9
	lift_min = 1.2
	length_min = 2

	CDC_records = []

	for i in range(0, len(binnedRate_CDC_Data.index)):
		CDC_records.append([str(binnedRate_CDC_Data.values[i,j]) for j in range(0, len(binnedRate_CDC_Data.columns))])

	#pprint(CDC_records)

	association_rules = apriori(CDC_records, min_support = support_min, min_confidence = confidence_min, length_min = 2)
	association_results = list(association_rules)
	print("association rules results: \n")
	#pprint(association_results)

	for item in association_results:
		pprint(str(item.items) + " support: " + str(item.support))

	#scatterPlot(binnedRate_CDC_Data['yearstart'], binnedRate_CDC_Data['datavalue'], "question", "rate", "rate by question", 0)

	
if __name__ == '__main__':
	main()


