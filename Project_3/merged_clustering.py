#merged data clustering
#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import colorbar
from scipy.cluster.hierarchy import dendrogram, linkage

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

import plotly.express as px
import plotly.graph_objs as go

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
			return 'very high'
		elif value <= 2.0 and value > 1.0:
			return 'high'
		elif value <= 1.0 and value > -1.0:
			return 'medium'
		elif value <= -1.0 and value > -2.0:
			return 'low'
		else:
			return  'very low'
	
	#apply the binning and create the new columns
	new_CDC_data[str(data_label + "_bin")] = new_CDC_data[data_label].apply(lambda x: categorization(x))
        
	return new_CDC_data

#makes a dendrogram for hierarchical clustering given linkage and labels
def makeDendrogram(linked, labelList):

	plt.clf()
	ax = plt.axes()
	dendrogram(linked,
	            orientation='top',
	            labels=labelList,
	            distance_sort='descending',
	            show_leaf_counts=True)

	plt.xlabel("Data Points")
	plt.ylabel("Height")
	ax.set_xticklabels([])
	ax.tick_params(which="both", bottom=False, left=True, labelsize = 10)

	plt.savefig('hierarchical clustering dendrogram.png')
	plt.clf()

# converts sorted values of 6 clusters averages to string representation
def pollution_categorization(value):
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

# creates cloropleth of the USA with states colored by assigned cluster
def usaMap2(loc, var, color, title, subtitle):
	# print('state abbrevations:' ,dataFrame[loc])
	# print('cluster label:' ,dataFrame[var])
	# exit()
	fig = go.Figure(data=go.Choropleth(
		locations = loc,
		z = var,
		locationmode = 'USA-states',
		colorscale = color,
	))

	fig.update_layout(
		title_text = title,
		geo_scope='usa',
		annotations = [dict(
        x=0,
        y=1,
        xref='paper',
        yref='paper',
        text=subtitle,
        showarrow = False
    )]
	)

	filename = 'usamap_'+title.replace(' ','_')+'.html'
	# fig.show()
	fig.write_html(filename, auto_open=True)

#creates a scatter plot and color codes the values by cluster labels if that parameter is passed in. Adds annotations to each data point as well.
def scatterPlot3(X_Data, Y_Data, labels, x_axis, y_axis, title, save, clusterLabels = None):
	
	plt.figure(1, figsize = (6,8))

	ax = plt.axes()


	if (clusterLabels != None):
		plt.scatter(X_Data, Y_Data, s=20, cmap = 'rainbow', c=clusterLabels)
	else:
		plt.scatter(X_Data, Y_Data, s=20)

	plt.title(title)
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)

	ax.tick_params(which="both", bottom=False, left=False, labelsize = 7)
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

	for i, state in enumerate(labels):
		x = X_Data[i]
		y = Y_Data[i]
		plt.text(x + 0.02, y + 0.02, state, fontsize = 9)

	if (save == True):
		plt.savefig(title + '.png')
		plt.clf()
	else:
		plt.show()
		plt.clf()

def findClusterAvg(new_clust_data):
	cluster_avg_pollution = [-1] * 6
	cluster_avg_cancer = [-1] * 6

	unique_cluster_labels = new_clust_data['cluster_labels'].unique()
	

	for cl_lbl in unique_cluster_labels:
		cluster_avg_pollution[cl_lbl] = (new_clust_data[new_clust_data['cluster_labels'] == cl_lbl]['AVG_REL_EST_TOTAL_PER_CAPITA'].mean())
		cluster_avg_cancer[cl_lbl] = (new_clust_data[new_clust_data['cluster_labels'] == cl_lbl]['AGE_ADJUSTED_CANCER_RATE'].mean())


	bins = ['v high', 'high', 'med high', 'med low', 'low', 'v low']

	cluster_avg_pollution_binned_sorted = sorted(cluster_avg_pollution, reverse=True)
	cluster_avg_pollution_binned = []
	for lbl in cluster_avg_pollution:
		x = cluster_avg_pollution_binned_sorted.index(lbl)
		val = bins[x]
		cluster_avg_pollution_binned.append(val)

	cluster_avg_cancer_binned_sorted = sorted(cluster_avg_cancer, reverse=True)
	cluster_avg_cancer_binned = []
	for lbl in cluster_avg_cancer:
		x = cluster_avg_cancer_binned_sorted.index(lbl)
		val = bins[x]
		cluster_avg_cancer_binned.append(val)
	
	final_cluster_labels = []
	for pollution_bin, cancer_bin in zip(cluster_avg_pollution_binned, cluster_avg_cancer_binned):
		s = pollution_bin + ", " + cancer_bin
		final_cluster_labels.append(s)

	annotation = "Avg:\n"
	i = 0
	for lbl in final_cluster_labels:
		annotation += "\t" + str(i) + "-[" + lbl + "]\n"
		i+=1

	print(annotation)
	
	return annotation
	# exit()

#driver code to generate visualizations and do the clustering (Hierarchical and KMeans) on the merged data set
def main():

	print("main")

	#preprocessing

	#read in merged_data2 that includes population and per-capita chemical release totals
	merged_data = pd.read_csv('merged_data2.csv' , sep=',', encoding='latin1')
	pprint(len(merged_data))

	#create a new dataframe with only year, state, chemicals per capita and cancer rate
	red_data = pd.concat([merged_data["YEAR"], merged_data["STATE_ABBR"], merged_data["AVG_REL_EST_TOTAL_PER_CAPITA"], merged_data["AGE_ADJUSTED_CANCER_RATE"]], axis = 1)

	#pprint(red_data)

	#copy data and rename series to save un-normalized data
	orig_chemicals = red_data['AVG_REL_EST_TOTAL_PER_CAPITA']
	orig_chemicals.dropna(inplace = True)
	orig_chemicals.rename("AVG_REL_EST_TOTAL_PER_CAPITA_ORIG", inplace = True)

	orig_cancer = red_data['AGE_ADJUSTED_CANCER_RATE']
	orig_cancer.dropna(inplace = True)
	orig_cancer.rename("AGE_ADJUSTED_CANCER_RATE_ORIG", inplace = True)


	#add column for the timezone region of the states
	region_data = separateByRegion(red_data, 'STATE_ABBR', True)

	#normalize the per capita chemical release totals and the cancer rate relative to all the values in each specific year
	norm_data = normalizeCDC_byQuestion(region_data, "YEAR",'AVG_REL_EST_TOTAL_PER_CAPITA')
	norm_data = normalizeCDC_byQuestion(norm_data, "YEAR",'AGE_ADJUSTED_CANCER_RATE')

	#drop rows with empty columns
	norm_data.dropna(inplace = True)

	#pprint(norm_data)
	cluster_data = norm_data.copy()

	#bin the rate of chemicals and cancer based on z-score and add columns for that
	binned_data = binRate(norm_data, 'AVG_REL_EST_TOTAL_PER_CAPITA')  #now datavalue_level is the bin label
	binned_data = binRate(norm_data, 'AGE_ADJUSTED_CANCER_RATE')  #now datavalue_level is the bin label

	#save the region and year series as  they will be dropped for preprocessing
	region_series = norm_data['region']
	year_series = norm_data["YEAR"]

	#pprint(cluster_data)

	#preprocess the data vieo label encoding categorical values (state only here)
	processed_data,  orig_data  = CDCpreprocessing2(cluster_data, ['STATE_ABBR'], ['YEAR', 'region'])    #(CDC_Data, categories, columns_to_drop):    new_CDC_Data, old_columns_CDC_Data

	#add back in the original, un-normalized series
	dict_names = {}
	for column in orig_data.columns:
		dict_names[column] = (column + "_Orig")

	orig_data.rename(columns=dict_names, inplace=True)

	#-----------------------------------------------------------------------------------------
	#making dataframe for final clustering

	#make a new data frame for clustering values
	new_clust_df = pd.DataFrame()

	#add state to dataframe from preprocessed data frame
	new_clust_df["STATE_ABBR"] = processed_data["STATE_ABBR"].unique()

	#get all unique states
	state_series = new_clust_df["STATE_ABBR"].unique()

	state_avg_cancer = []
	state_avg_chem = []

	#add in the original un-normalized data and the years
	processed_data = pd.concat([processed_data, orig_cancer, orig_chemicals, year_series], axis = 1)

	#pprint(processed_data)

	#drop rows with empty data
	processed_data.dropna(inplace = True)
	pprint(len(processed_data))

	#calculate the average chemical release and cancer rate for each state over time and append to lists
	for state in processed_data["STATE_ABBR"].unique():
		state_avg_cancer.append( processed_data.loc[processed_data['STATE_ABBR'] == state]['AGE_ADJUSTED_CANCER_RATE_ORIG'].mean() )
		state_avg_chem.append( processed_data.loc[processed_data['STATE_ABBR'] == state]['AVG_REL_EST_TOTAL_PER_CAPITA_ORIG'].mean() )

	#pprint(year_series)

	#add new columns in dataframe for average chemicals and average cancer rate
	new_clust_df["AGE_ADJUSTED_CANCER_RATE"] = state_avg_cancer
	new_clust_df["AVG_REL_EST_TOTAL_PER_CAPITA"] = state_avg_chem

	#do hierarchical and Kmeans clustering
	silh_score_list = []
	cluster_vals = []

	pprint(new_clust_df.columns)

	new_clust_data, silh_score = doHierarchical(new_clust_df, 6)    #(CDC_Data, k):   new_CDC_data, silhouette_avg
	print("silhouette score: ", silh_score)

	new_clust_data2, silh_score = doKMeans(new_clust_df, 6)    #(CDC_Data, k):   new_CDC_data, silhouette_avg
	print("silhouette score: ", silh_score)

	#get the unique state abbreviations from original data (before label encoding)
	state_data = pd.Series(orig_data['STATE_ABBR_Orig'].unique())
	state_data.rename('STATE_ABBR_Orig', inplace = True)

	#add state abbreviation back into the dataframes
	new_clust_data = pd.concat([new_clust_data, state_data], axis = 1)
	new_clust_data2 = pd.concat([new_clust_data2, state_data], axis = 1)

	#save the cluster labels as lists
	cluster_labels_list2 = new_clust_data['cluster_labels'].astype(int).tolist()
	cluster_labels_list3 = new_clust_data2['cluster_labels'].astype(int).tolist()

	pprint(new_clust_data)
	pprint(len(new_clust_data))

	#make scatter plots with the clusters
	scatterPlot3(new_clust_data['AGE_ADJUSTED_CANCER_RATE'], new_clust_data['AVG_REL_EST_TOTAL_PER_CAPITA'], new_clust_data['STATE_ABBR_Orig'], "Average Age Adjusted Cancer Rate (per 100,000 people)", "Average Chemical Release Estimate Per Capita", "Hierarchical Clustering", True, cluster_labels_list2)
	scatterPlot3(new_clust_data2['AGE_ADJUSTED_CANCER_RATE'], new_clust_data2['AVG_REL_EST_TOTAL_PER_CAPITA'], new_clust_data['STATE_ABBR_Orig'], "Average Age Adjusted Cancer Rate (per 100,000 people)", "Average Chemical Release Estimate Per Capita", "KMeans Clustering", True, cluster_labels_list3)

	#make cloropleth of clusters
	cluster_averages = findClusterAvg(new_clust_data)

	usaMap2(norm_data["STATE_ABBR"].unique(), new_clust_data['cluster_labels'], 'rainbow', "Hierarchical Clusters (n = 6), Cluster Avg Format = (pollution, cancer)", cluster_averages)

	#----------------------------------------

	#drop the state abbreviation for dendrogram production
	new_clust_data.drop(['STATE_ABBR_Orig'], axis = 1, inplace = True)

	#make the dendrogram for the hierarchical clustering
	makeDendrogram(linkage(new_clust_data, 'single'), range(0, len(new_clust_data["AVG_REL_EST_TOTAL_PER_CAPITA"])))

if __name__ == '__main__':
	main()