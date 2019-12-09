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

#function to generate the final network data frame
def getWeights(final_network, network_df, normalized_data):

	#weight factors for edges
	region_weight = 0.1
	cancer_weight = 0.3
	chemical_weight = 0.3
	correl_cancer_weight = 0.15
	correl_chem_weight = 0.15

	final_weights = []

	#calculate weights and append to the list of edge weights
	for index, row in final_network.iterrows():   #for each row in the final_network df, calculate the weights
		weight = 0.0
		if(   network_df.iloc[row['Src']]['REGION']  ==  network_df.iloc[row['Dst']]['REGION']  ):	#if they have the same region   
			weight += region_weight
		else:
			weight += 0.0

		if(   network_df.iloc[row['Src']]['CANCER_LEVEL']  ==  network_df.iloc[row['Dst']]['CANCER_LEVEL']  ):	#if they have the same cancer level   
			weight += cancer_weight
		else:
			weight += 0.0

		if(   network_df.iloc[row['Src']]['CHEMICAL_LEVEL']  ==  network_df.iloc[row['Dst']]['CHEMICAL_LEVEL']  ):	#if they have the same chemicals level   
			weight += chemical_weight
		else:
			weight += 0.0
 
		
		state1 = network_df.iloc[ row['Src'] ]['STATE_ABBR']  #state for source 
		state2 = network_df.iloc[ row['Dst'] ]['STATE_ABBR']  #state for destination
		print(state1, state2)

		#get the two cancer rate series for the two states 		
		series1 = normalized_data.loc[normalized_data["STATE_ABBR"] == state1]['AGE_ADJUSTED_CANCER_RATE']
		series2 = normalized_data.loc[normalized_data["STATE_ABBR"] == state2]['AGE_ADJUSTED_CANCER_RATE']

		correl = 0.000

		#adjust size of series if they are different due to missing values
		if (series1.size != series2.size):  #different lengths

			if (series1.size < series2.size):
				series2 = series2[(series2.size - series1.size):]
			else:
				series1 = series1[(series1.size - series2.size):]

		try:
			#calculate the correlation over time for the two state combination for the cancer rate
			linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			correl = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			#print('coefficient of determination:' + state1 + " " + state2, r_sq)
		except:
			print("err")
			correl = 0.0

		print(correl)

		#add the weight for the correlation to the final weight
		weight += correl*correl_cancer_weight

		#get the chemical release estimate series for the two states 
		series1 = normalized_data[normalized_data["STATE_ABBR"] == state1]['AVG_REL_EST_TOTAL_PER_CAPITA']
		series2 = normalized_data[normalized_data["STATE_ABBR"] == state2]['AVG_REL_EST_TOTAL_PER_CAPITA']

		correl = 0.00

		#adjust size of series if they are different due to missing values
		if (series1.size != series2.size):  #different lengths
			if (series1.size < series2.size):
				series2 = series2[(series2.size - series1.size):]
			else:
				series1 = series1[(series1.size - series2.size):]

		try:
			#calculate the correlation over time for the two states
			linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			correl = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			#print('coefficient of determination:' + state1 + " " + state2, r_sq)
		except:
			correl = 0.0

		#add the weight for the correlation to the final weight
		weight += correl*correl_chem_weight

		#print(weight)

		#append final weight to the list of weights
		final_weights.append( round(weight,4)  )  

	return final_weights


#drivere program to generate the final data frame with edges and weights for the network
def main():
	print("main")

	#read in the merged data set with the per capita chemical release estimates
	merged_data = pd.read_csv('merged_data2.csv' , sep=',', encoding='latin1')

	#create a copy of the dataframe and add the timezone region labels
	new_data = merged_data.copy()
	region_data = separateByRegion(new_data)

	#drop all extra column except region, year, state, cancer and chemicals
	region_data = region_data.loc[:, region_data.columns.intersection(['YEAR', 'REGION', 'STATE_ABBR', 'AVG_REL_EST_TOTAL_PER_CAPITA', 'AGE_ADJUSTED_CANCER_RATE'])]
	
	#drop rows in empty values
	region_data.dropna(inplace = True)
	pprint(len(region_data))

	#save the cancer rate before normalization by z-score and rename
	cancer_series = region_data["AGE_ADJUSTED_CANCER_RATE"]
	cancer_series.rename("AGE_ADJUSTED_CANCER_RATE_ORIG", inplace = True)

	chemical_series = region_data["AVG_REL_EST_TOTAL_PER_CAPITA"]
	chemical_series.rename("AVG_REL_EST_TOTAL_PER_CAPITA_ORIG", inplace = True)

	#normalize the cancer and chemical release estimate values
	normalized_data = normalize_byQuestion(region_data, 'YEAR', 'AGE_ADJUSTED_CANCER_RATE')
	normalized_data = normalize_byQuestion(normalized_data, 'YEAR', 'AVG_REL_EST_TOTAL_PER_CAPITA')

	#add back in the original un-normalized chemical release and cancer rate data
	normalized_data = pd.concat([normalized_data, cancer_series, chemical_series], axis = 1)

	#rename columns to represent the values in them
	normalized_data.rename(columns={"AGE_ADJUSTED_CANCER_RATE": "AGE_ADJUSTED_CANCER_RATE_Z", "AVG_REL_EST_TOTAL_PER_CAPITA": "AVG_REL_EST_TOTAL_Z"}, inplace = True)
	normalized_data.rename(columns={"AGE_ADJUSTED_CANCER_RATE_ORIG": "AGE_ADJUSTED_CANCER_RATE", "AVG_REL_EST_TOTAL_PER_CAPITA_ORIG": "AVG_REL_EST_TOTAL_PER_CAPITA"}, inplace = True)

	#pprint(normalized_data)

	#create a dataframe for the network
	network_df = pd.DataFrame()

	#sort the data by state and then year
	normalized_data.sort_values(by=['STATE_ABBR', 'YEAR'], inplace = True)

	#bin the cancer rate and chemical release estimate by z-score and add columns for that
	binned_data = binRate(normalized_data, 'AVG_REL_EST_TOTAL_Z')  #now columnname_bin
	binned_data = binRate(binned_data, 'AGE_ADJUSTED_CANCER_RATE_Z')  #now columnname_bin

	pprint(binned_data)

	#for the network dataframe, get the unique state values and create a columns for that
	network_df['STATE_ABBR'] = normalized_data["STATE_ABBR"].unique()  #alphabetical for states

	#add a column for the timezone region labels
	for state in network_df['STATE_ABBR']:
		network_df['REGION'] = network_df['STATE_ABBR'].apply(lambda x: categorization(x))

	#pprint(network_df)

	#in the network dataframe, add columns for cancer and chemicals level by the most common value for that over the year for each state
	mode_chemical = []
	mode_cancer = []

	for state in network_df['STATE_ABBR'].unique():
		mode_chemical.append( normalized_data[normalized_data['STATE_ABBR'] == state]['AVG_REL_EST_TOTAL_Z_bin'].mode().iat[0] )
		mode_cancer.append( normalized_data[normalized_data['STATE_ABBR'] == state]['AGE_ADJUSTED_CANCER_RATE_Z_bin'].mode().iat[0] )

	network_df['CHEMICAL_LEVEL'] = mode_chemical
	network_df['CANCER_LEVEL'] = mode_cancer

	network_df.to_csv("network_df.csv") 

	#pprint(network_df)

	#--------------------------------------

	#create a final network dataframe which is essentially the edges
	final_network = pd.DataFrame( 
		data=list(combinations(network_df.index.tolist(), 2)), 
		columns=['Src', 'Dst'])

	#get all the state abbreviations and map them to the numerical number (index) that thehy correspond to 
	states_df = pd.concat([network_df["STATE_ABBR"]], axis = 1)
	states_dict = states_df.to_dict('index')
	new_states_dict = {}   #dictionary
	counter = 0
	for item in states_dict:
		new_states_dict[counter] = states_dict.get(item).get("STATE_ABBR")
		counter += 1

	#get the list of weights for each edge
	final_weights = getWeights(final_network, network_df, normalized_data)
	#print(final_weights)

	#add a column for the edge weights
	final_network['Wght'] = final_weights

	#replace the state numbers with the state abbreviations
	final_network.replace(new_states_dict, inplace = True)

	pprint(final_network)

	#save the final network of edges and weights to a .csv file
	final_network.to_csv("final_network.csv",index=False)   #has full connectivity

if __name__ == '__main__':
	main()
