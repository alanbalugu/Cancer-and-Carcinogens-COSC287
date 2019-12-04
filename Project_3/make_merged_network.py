#network of states
#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint

from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn import metrics

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)

#imports from other files
from CdcClustering import normalizeCDC   #(CDC_Data, columns_to_norm):   CDC_data_norm
from CdcClustering import scatterPlot    #(X_Data, Y_Data, x_axis, y_axis, title, save):
from CdcClustering import scatterPlot2

from CdcClustering import normalizeCDC_byQuestion   #(CDC_data, feature_to_sort_by, feature_to_clean):
from CdcClustering import doPCA   #(CDC_Data)

from AssociationRuleMining import categorization
from AssociationRuleMining import abbreviate

from merged_heatmaps import stateWideCorrel  #(region_data (sorted), 'AGE_ADJUSTED_CANCER_RATE'):    lin_reg_matrix
from merged_heatmaps import separateByRegion #(CDC_Data):   data   (REGION, STATE_ABBR)
from merged_heatmaps import separate_by_year  #(CDC_Data, year):   data   (YEAR)
from merged_heatmaps import doLinearRegr   #(series1, series2):   r_sq

from CdcClustering import normalizeCDC_byQuestion as normalize_byQuestion #(CDC_data, feature_to_sort_by, feature_to_clean):
from merged_clustering import binRate #binRate(CDC_Data, data_label):   dataframe


def getWeights(final_network, network_df, normalized_data):
	region_weight = 0.1
	cancer_weight = 0.3
	chemical_weight = 0.3

	correl_cancer_weight = 0.15
	correl_chem_weight = 0.15

	final_weights = []

	#0   1
	for index, row in final_network.iterrows():   #for each row in the final_network df, calculate the weights
		weight = 0.0
		if(   network_df.iloc[row['Src']]['REGION']  ==  network_df.iloc[row['Dst']]['REGION']  ):			#if they have the same region   
			weight += region_weight
		else:
			weight += 0.0

		if(   network_df.iloc[row['Src']]['CANCER_LEVEL']  ==  network_df.iloc[row['Dst']]['CANCER_LEVEL']  ):			#if they have the same region   
			weight += cancer_weight
		else:
			weight += 0.0

		if(   network_df.iloc[row['Src']]['CHEMICAL_LEVEL']  ==  network_df.iloc[row['Dst']]['CHEMICAL_LEVEL']  ):			#if they have the same region   
			weight += chemical_weight
		else:
			weight += 0.0

		#series1 = merged_data.loc[merged_data['STATE_ABBR'] == state1][data_label]

		# 0,1    0, 2   0, 3   ->  50,0   50, 1   
		
		state1 = network_df.iloc[ row['Src'] ]['STATE_ABBR']  #state for source   source gives a number. iloc of that num gives a dataframe.  get state abbr string 
		state2 = network_df.iloc[ row['Dst'] ]['STATE_ABBR']  #state for destination
		print(state1, state2)
 		#calculate the correlatino over time for the two state combination

		series1 = normalized_data.loc[normalized_data["STATE_ABBR"] == state1]['AGE_ADJUSTED_CANCER_RATE']
		series2 = normalized_data.loc[normalized_data["STATE_ABBR"] == state2]['AGE_ADJUSTED_CANCER_RATE']

		#print(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))

		correl = 0.000

		if (series1.size != series2.size):  #different lengths

			if (series1.size < series2.size):
				series2 = series2[(series2.size - series1.size):]
			else:
				series1 = series1[(series1.size - series2.size):]

		try:
			linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			correl = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			#print('coefficient of determination:' + state1 + " " + state2, r_sq)
		except:
			print("err")
			correl = 0.0

		print(correl)
		weight += correl*correl_cancer_weight

		series1 = normalized_data[normalized_data["STATE_ABBR"] == state1]['AVG_REL_EST_TOTAL']
		series2 = normalized_data[normalized_data["STATE_ABBR"] == state2]['AVG_REL_EST_TOTAL']

		correl = 0.00

		if (series1.size != series2.size):  #different lengths
			if (series1.size < series2.size):
				series2 = series2[(series2.size - series1.size):]
			else:
				series1 = series1[(series1.size - series2.size):]

		try:
			linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))

			correl = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			#print('coefficient of determination:' + state1 + " " + state2, r_sq)
		except:
			correl = 0.0

		weight += correl*correl_chem_weight

		#print(weight)
		final_weights.append( round(weight,4)  )  #scale weight by 5 for greater separation

	return final_weights


def main():
	print("main")

	merged_data = pd.read_csv('merged_data.csv' , sep=',', encoding='latin1')

	new_data = merged_data.copy()
	region_data = separateByRegion(new_data)  #adds region column

	region_data = region_data.loc[:, region_data.columns.intersection(['YEAR', 'REGION', 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE'])]
	region_data.dropna(inplace = True)

	cancer_series = region_data["AGE_ADJUSTED_CANCER_RATE"]
	chemical_series = region_data["AVG_REL_EST_TOTAL"]

	region_data.rename(columns={"AGE_ADJUSTED_CANCER_RATE": "AGE_ADJUSTED_CANCER_RATE_ORIG", "AVG_REL_EST_TOTAL": "AVG_REL_EST_TOTAL_ORIG"}, inplace = True)

	normalized_data = normalize_byQuestion(region_data, 'YEAR', 'AGE_ADJUSTED_CANCER_RATE_ORIG')
	normalized_data = normalize_byQuestion(normalized_data, 'YEAR', 'AVG_REL_EST_TOTAL_ORIG')

	normalized_data = pd.concat([normalized_data, cancer_series, chemical_series], axis = 1)

	normalized_data.rename(columns={"AGE_ADJUSTED_CANCER_RATE": "AGE_ADJUSTED_CANCER_RATE", "AVG_REL_EST_TOTAL": "AVG_REL_EST_TOTAL"}, inplace = True)
	normalized_data.rename(columns={"AGE_ADJUSTED_CANCER_RATE_ORIG": "AGE_ADJUSTED_CANCER_RATE_Z", "AVG_REL_EST_TOTAL_ORIG": "AVG_REL_EST_TOTAL_Z"}, inplace = True)


	#pprint(normalized_data)

	# 51 by 51 matrices (50 states + DC)
	#cancer_correlations_matrix = stateWideCorrel(normalized_data, 'AGE_ADJUSTED_CANCER_RATE')  #(region_data (sorted), 'AGE_ADJUSTED_CANCER_RATE'):    lin_reg_matrix
	#chemical_correlations_matrix = stateWideCorrel(normalized_data, 'AVG_REL_EST_TOTAL')  #(region_data (sorted), 'AGE_ADJUSTED_CANCER_RATE'):    lin_reg_matrix

	network_df = pd.DataFrame()

	#things we want in the network:  shared region?  shared cancer level?  shared chemical level?  by time-wise correlation?
	#weightage:   0.1-region   0.4-cancer   0.4-chemical  0.1-time correlation
	#things to calculate:   correlations between states

	#compare each state to the other state

	normalized_data.sort_values(by=['STATE_ABBR', 'YEAR'], inplace = True)

	binned_data = binRate(normalized_data, 'AVG_REL_EST_TOTAL_Z')  #now columnname_bin
	binned_data = binRate(binned_data, 'AGE_ADJUSTED_CANCER_RATE_Z')  #now columnname_bin

	#pprint(binned_data)

	network_df['STATE_ABBR'] = normalized_data["STATE_ABBR"].unique()  #alphabetical for states

	for state in network_df['STATE_ABBR']:
		network_df['REGION'] = network_df['STATE_ABBR'].apply(lambda x: categorization(x))

	#pprint(network_df)
	mode_chemical = []
	mode_cancer = []

	for state in network_df['STATE_ABBR'].unique():
		mode_chemical.append( normalized_data[normalized_data['STATE_ABBR'] == state]['AVG_REL_EST_TOTAL_Z_bin'].mode().iat[0] )
		mode_cancer.append( normalized_data[normalized_data['STATE_ABBR'] == state]['AGE_ADJUSTED_CANCER_RATE_Z_bin'].mode().iat[0] )

	network_df['CHEMICAL_LEVEL'] = mode_chemical
	network_df['CANCER_LEVEL'] = mode_cancer

	#pprint(network_df)

	#--------------------------------------

	final_network = pd.DataFrame( 
		data=list(combinations(network_df.index.tolist(), 2)), 
		columns=['Src', 'Dst'])

	final_weights = getWeights(final_network, network_df, normalized_data)
	print(final_weights)

	final_network['Wght'] = final_weights

	pprint(final_network)

	final_network.to_csv("final_network.csv",index=False)   #has full connectivity

	#iterate through matrix and append values to a list

if __name__ == '__main__':
	main()
