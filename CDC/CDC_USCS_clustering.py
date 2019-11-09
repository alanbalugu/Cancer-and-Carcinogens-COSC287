#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import colorbar


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
import pylab as plt
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

#imports for preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from scipy.stats import ttest_ind

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)

#imports from other files
from CdcClustering import normalizeCDC   #(CDC_Data, columns_to_norm):   CDC_data_norm
from CdcClustering import scatterPlot    #(X_Data, Y_Data, x_axis, y_axis, title, save):
from CdcClustering import scatterPlot2
from CdcClustering import cycleClustering    #(CDC_data, type_clus, range_list, silh_score_list, cluster_vals):   clustered_CDC_data, silh_score_list, cluster_vals

from CdcClustering import separate_by_year  #(CDC_Data, year):

from CdcClustering import normalizeCDC_byQuestion   #(CDC_data, feature_to_sort_by, feature_to_clean):
from CdcClustering import doDBScan   #(CDC_Data, nearness, min_samples):   new_CDC_data, silhouette_avg, n_clusters_
from CdcClustering import doKMeans    #(CDC_Data, k):   new_CDC_data, silhouette_avg
from CdcClustering import doHierarchical    #(CDC_Data, k):   new_CDC_data, silhouette_avg
from CdcClustering import CDCpreprocessing   #(CDC_Data, categories, columns_to_drop):	new_CDC_data, (series) list_of_dropped, list_of_categ
from CdcClustering import scatterPlot     #(X_Data, Y_Data, x_axis, y_axis, title, save):
from CdcClustering import cycleClustering    #(CDC_data, type_clus, range_list, silh_score_list, cluster_vals):     clustered_CDC_data, silh_score_list, cluster_vals (cluster k or eps)
from CdcClustering import CDCpreprocessing2    #(CDC_Data, categories, columns_to_drop):    new_CDC_Data, old_columns_CDC_Data
from CdcClustering import doPCA   #(CDC_Data)

from AssociationRuleMining import categorization
from AssociationRuleMining import abbreviate
from AssociationRuleMining import doAssociationRuleMining

#separates the cdc data by the column for the state and returns the dataframe with the region column based on state timezone. Abbreviates state name if necessary
def separateByRegion(CDC_Data, state_label, has_abbrev):

	new_CDC_data = CDC_Data

	if (has_abbrev == False):

		new_CDC_data[state_label] = new_CDC_data[state_label].apply(lambda x: abbreviate(x))

		new_CDC_data['region'] = new_CDC_data[state_label].apply(lambda x: categorization(x))
	        
		return new_CDC_data

	else:
		new_CDC_data['region'] = new_CDC_data[state_label].apply(lambda x: categorization(x))
		return new_CDC_data

#adds a column for the binned level for the data based on the z-score. 
def binRate(CDC_Data, data_label):
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
	
	#apply the binning and create the new columns
	new_CDC_data['datavalue_level'] = new_CDC_data[data_label].apply(lambda x: categorization(x))
        
	return new_CDC_data

#driver program to run the clustering on the cancer trends over time data
def main():
	print("main")

	CDC_USCS_data = pd.read_csv('USCS_CancerTrends_OverTime_ByState.csv' , sep=',', encoding='latin1')
	CDC_USCS_data.dropna(inplace = True)

	region_CDC_data = separateByRegion(CDC_USCS_data, 'Area', False)

	normalized_CDC_data = normalizeCDC(region_CDC_data, ['AgeAdjustedRate'])

	binned_CDC_data = binRate(normalized_CDC_data, 'AgeAdjustedRate')  #now datavalue_level is the bin label

	#data now has region solumn and rate binned into another column as high low etc
	#lets do association rule mining:

	#doAssociationRuleMining(binned_CDC_data)

	print("----------------------------------------")
	#lets do clustering now

	categ_columns = ['Area', 'region', 'datavalue_level']
	drop_columns = ['CancerType', 'lci', 'uci', 'CaseCount', 'Population'] #remove columns that are unnecessary for clustering question

	silh_score_list = []
	cluster_vals = []

	processed_CDC_data, old_columns_CDC_Data = CDCpreprocessing2(binned_CDC_data, categ_columns, drop_columns)   #(CDC_Data, categories, columns_to_drop):    new_CDC_data, (series) list_of_dropped, list_of_categ
	#old data is a data frame of the original data that was label encoded

	#add back in the original, un encoded data for diaplying purposes (after clustering)
	print(old_columns_CDC_Data.columns)
	dict_names = {}
	for column in old_columns_CDC_Data.columns:
		dict_names[column] = (column + "Orig")

	old_columns_CDC_Data.rename(columns=dict_names, inplace=True)

	cluster_type = "K"
	clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_CDC_data, cluster_type, np.arange(30, 60, 1), silh_score_list, cluster_vals)
	#clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_CDC_data, "H", np.arange(30, 60, 1), silh_score_list, cluster_vals)
	#clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_CDC_data, "D", np.arange(0.3, 7.0, 0.2), silh_score_list, cluster_vals)

	added_CDC_Data = clustered_CDC_data
	added_CDC_Data = pd.concat([added_CDC_Data, old_columns_CDC_Data], axis = 1)

	added_CDC_Data.dropna(inplace = True)

	#convert into integers
	cluster_labels_list = added_CDC_Data['cluster_labels'].astype(int).tolist()
	print(cluster_labels_list)

	#plot the clusters with differet axes to visualize clustering. Plots the normalized cancer rate by different variables
	scatterPlot2(added_CDC_Data['Year'], added_CDC_Data['AgeAdjustedRate'], "year", "rate", "rate by year "+cluster_type, True, cluster_labels_list)
	scatterPlot2(added_CDC_Data['AreaOrig'], added_CDC_Data['AgeAdjustedRate'], "state", "rate", "rate by state", True, cluster_labels_list)
	scatterPlot2(added_CDC_Data['regionOrig'], added_CDC_Data['AgeAdjustedRate'], "region", "rate", "rate by region", True, cluster_labels_list)
	scatterPlot2(added_CDC_Data['datavalue_level'], added_CDC_Data['AgeAdjustedRate'], "rate bin", "rate", "rate by rate bin", True, cluster_labels_list)

	#do the PCA to see variation
	doPCA(clustered_CDC_data)

if __name__ == '__main__':
	main()


