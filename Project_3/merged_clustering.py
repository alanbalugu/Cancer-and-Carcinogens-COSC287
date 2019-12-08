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

# PLOTLY
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
	new_CDC_data[str(data_label + "_bin")] = new_CDC_data[data_label].apply(lambda x: categorization(x))
		
	return new_CDC_data

def makeDendrogram(linked, labelList):
	#linked = linkage(processed_data, 'single')
	#labelList = range(0, len(processed_data["AVG_REL_EST_TOTAL"]))
	plt.clf()
	#plt.figure(figsize=(10, 7))
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

	plt.show()


#creates a scatter plot and color codes the values by cluster labels if that parameter is passed in
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

# def usaMap(dataFrame, loc, var, color, title):
#     fig = go.Figure(data=go.Choropleth(
#         locations=dataFrame[loc],
#         z = dataFrame[var],
#         locationmode = 'USA-states',
#         colorscale = color,
#     ))

#     fig.update_layout(
#         title_text = title,
#         geo_scope='usa',
#     )

#     filename = 'usamap_'+title.replace(' ','_')+'.html'
#     fig.write_html(filename, auto_open=True)

def usaMap2(loc, var, color, title):
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
	)

	filename = 'usamap_'+title.replace(' ','_')+'.html'
	fig.show()
	fig.write_html(filename, auto_open=True)


def main():

	print("main")

	merged_data = pd.read_csv('merged_data2.csv' , sep=',', encoding='latin1')

	red_data = pd.concat([merged_data["YEAR"], merged_data["STATE_ABBR"], merged_data["AVG_REL_EST_TOTAL_PER_CAPITA"], merged_data["AGE_ADJUSTED_CANCER_RATE"]], axis = 1)

	pprint(red_data)

	orig_chemicals = red_data['AVG_REL_EST_TOTAL_PER_CAPITA']
	orig_chemicals.dropna(inplace = True)
	orig_chemicals.rename("AVG_REL_EST_TOTAL_PER_CAPITA_ORIG", inplace = True)

	orig_cancer = red_data['AGE_ADJUSTED_CANCER_RATE']
	orig_cancer.dropna(inplace = True)
	orig_cancer.rename("AGE_ADJUSTED_CANCER_RATE_ORIG", inplace = True)

	region_data = separateByRegion(red_data, 'STATE_ABBR', True)

	norm_data = normalizeCDC_byQuestion(region_data, "YEAR",'AVG_REL_EST_TOTAL_PER_CAPITA')
	norm_data = normalizeCDC_byQuestion(norm_data, "YEAR",'AGE_ADJUSTED_CANCER_RATE')

	norm_data.dropna(inplace = True)

	#pprint(norm_data)
	cluster_data = norm_data.copy()

	binned_data = binRate(norm_data, 'AVG_REL_EST_TOTAL_PER_CAPITA')  #now datavalue_level is the bin label
	binned_data = binRate(norm_data, 'AGE_ADJUSTED_CANCER_RATE')  #now datavalue_level is the bin label

	region_series = norm_data['region']

	year_series = norm_data["YEAR"]

	#pprint(cluster_data)

	processed_data,  orig_data  = CDCpreprocessing2(cluster_data, ['STATE_ABBR'], ['YEAR', 'region'])    #(CDC_Data, categories, columns_to_drop):    new_CDC_Data, old_columns_CDC_Data

	pprint(processed_data)

	cluster_type = "H"

	silh_score_list = []
	cluster_vals = []

	pprint(processed_data.columns)
	#clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_data, "K", np.arange(15, 35, 1), silh_score_list, cluster_vals)
	clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_data, "H", np.arange(15, 20, 1), silh_score_list, cluster_vals)
	#clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_data, "D", np.arange(0.3, 7.0, 0.2), silh_score_list, cluster_vals)

	dict_names = {}
	for column in orig_data.columns:
		dict_names[column] = (column + "_Orig")

	orig_data.rename(columns=dict_names, inplace=True)

	added_CDC_Data = clustered_CDC_data
	added_CDC_Data = pd.concat([added_CDC_Data, orig_data], axis = 1)
	added_CDC_Data = pd.concat([added_CDC_Data, year_series, region_series], axis = 1)
	added_CDC_Data = pd.concat([added_CDC_Data, orig_cancer, orig_chemicals], axis = 1)
	added_CDC_Data.dropna(inplace = True)

	#convert into integers
	cluster_labels_list = added_CDC_Data['cluster_labels'].astype(int).tolist()
	#print(cluster_labels_list)
	#pprint(added_CDC_Data)

	#pprint(added_CDC_Data)

	#plot the clusters with differet axes to visualize clustering. Plots the normalized cancer rate by different variables
	#scatterPlot(cluster_vals, silh_score_list, "Cluster Size", "Silhouette Score",'silhouette score by cluster size '+cluster_type, True)
	#scatterPlot2(added_CDC_Data['YEAR'], added_CDC_Data['AGE_ADJUSTED_CANCER_RATE_ORIG'], "Year", "Age Adjusted Cancer Rate (per 100,000 people)", "cancer rate by year "+cluster_type, True, cluster_labels_list)
	#scatterPlot2(added_CDC_Data['region'], added_CDC_Data['AGE_ADJUSTED_CANCER_RATE_ORIG'], "Timezone Region", "Age Adjusted Cancer Rate (per 100,000 people)", "cancer rate by region "+cluster_type, True, cluster_labels_list)
	#scatterPlot2(added_CDC_Data['AVG_REL_EST_TOTAL_PER_CAPITA_ORIG'], added_CDC_Data['AGE_ADJUSTED_CANCER_RATE_ORIG'], "Chemical Pollution Release Per Capita", "Age Adjusted Cancer Rate (per 100,000 people)", "cancer rate by pollution "+cluster_type, True, cluster_labels_list)
	#scatterPlot2(added_CDC_Data['STATE_ABBR_Orig'], added_CDC_Data['AVG_REL_EST_TOTAL_PER_CAPITA_ORIG'], "US States", "Average Chemical Release Estimate Per Capita", "rate by state "+cluster_type, True, cluster_labels_list)
	
	# fig = px.scatter_3d(x = added_CDC_Data['STATE_ABBR_Orig'], y = added_CDC_Data['AGE_ADJUSTED_CANCER_RATE_ORIG'], z = added_CDC_Data['AVG_REL_EST_TOTAL_PER_CAPITA_ORIG'], color = cluster_labels_list)
	# fig.show()

	#----------------------------------------------------------------

	new_Data, silh_score = doHierarchical(processed_data, 25)    #(CDC_Data, k):   new_CDC_data, silhouette_avg
	print("silhouette score: ", silh_score)

	#--------------------------------------

	new_clust_df = pd.DataFrame()

	new_clust_df["STATE_ABBR"] = processed_data["STATE_ABBR"].unique()

	state_series = new_clust_df["STATE_ABBR"].unique()

	state_avg_cancer = []
	state_avg_chem = []

	processed_data = pd.concat([processed_data, orig_cancer, orig_chemicals, year_series], axis = 1)

	year_series = []
	#pprint(processed_data)

	processed_data.dropna(inplace = True)

	for state in processed_data["STATE_ABBR"].unique():
		state_avg_cancer.append( processed_data.loc[processed_data['STATE_ABBR'] == state]['AGE_ADJUSTED_CANCER_RATE_ORIG'].mean() )
		state_avg_chem.append( processed_data.loc[processed_data['STATE_ABBR'] == state]['AVG_REL_EST_TOTAL_PER_CAPITA_ORIG'].mean() )
		#year_series.append(processed_data.loc[processed_data['STATE_ABBR'] == state]["YEAR"].unique())

	#pprint(year_series)

	new_clust_df["AGE_ADJUSTED_CANCER_RATE"] = state_avg_cancer
	new_clust_df["AVG_REL_EST_TOTAL_PER_CAPITA"] = state_avg_chem

	silh_score_list = []
	cluster_vals = []

	#typeCluster = ""
	new_cluster_data_hierarchical, silh_score = doHierarchical(new_clust_df, 6)    #(CDC_Data, k):   new_CDC_data, silhouette_avg
	new_clust_data_k_means, silh_score = doKMeans(new_clust_df, 6)    #(CDC_Data, k):   new_CDC_data, silhouette_avg

	print("silhouette score: ", silh_score)

	state_data = pd.Series(orig_data['STATE_ABBR_Orig'].unique())
	state_data.rename('STATE_ABBR_Orig', inplace = True)

	#new_cluster_data_hierarchical['YEAR'] = year_series

	new_cluster_data_hierarchical = pd.concat([new_cluster_data_hierarchical, state_data], axis = 1)
	new_clust_data_k_means = pd.concat([new_clust_data_k_means, state_data], axis = 1)

	cluster_labels_hier = new_cluster_data_hierarchical['cluster_labels'].astype(int).tolist()
	cluster_labels_kmeans = new_clust_data_k_means['cluster_labels'].astype(int).tolist()

	# pprint(new_cluster_data_hierarchical)
	
	scatterPlot3(new_cluster_data_hierarchical['AGE_ADJUSTED_CANCER_RATE'], new_cluster_data_hierarchical['AVG_REL_EST_TOTAL_PER_CAPITA'], new_cluster_data_hierarchical['STATE_ABBR_Orig'], "Average Age Adjusted Cancer Rate (per 100,000 people)", "Average Chemical Release Estimate Per Capita", "cool clusters" + "H", True, cluster_labels_hier)
	scatterPlot3(new_clust_data_k_means['AGE_ADJUSTED_CANCER_RATE'], new_clust_data_k_means['AVG_REL_EST_TOTAL_PER_CAPITA'], new_cluster_data_hierarchical['STATE_ABBR_Orig'], "Average Age Adjusted Cancer Rate (per 100,000 people)", "Average Chemical Release Estimate Per Capita", "cool clusters" + "K", True, cluster_labels_kmeans)
	

	usaMap2(norm_data["STATE_ABBR"].unique(), new_clust_data_k_means['cluster_labels'], 'rainbow', "K Means Clusters (n = 6)")

	# #clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_data, "K", np.arange(15, 35, 1), silh_score_list, cluster_vals)
	# clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_data, "H", np.arange(15, 35, 1), silh_score_list, cluster_vals)
	# #clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(processed_data, "D", np.arange(0.3, 7.0, 0.2), silh_score_list, cluster_vals)

	# dict_names = {}
	# for column in orig_data.columns:
	# 	dict_names[column] = (column + "_Orig")

	# orig_data.rename(columns=dict_names, inplace=True)

	# added_CDC_Data = clustered_CDC_data
	# added_CDC_Data = pd.concat([added_CDC_Data, orig_data], axis = 1)
	# added_CDC_Data = pd.concat([added_CDC_Data, year_series, region_series], axis = 1)
	# added_CDC_Data = pd.concat([added_CDC_Data, orig_cancer, orig_chemicals], axis = 1)
	# added_CDC_Data.dropna(inplace = True)

	#----------------------------------------

	new_cluster_data_hierarchical.drop(['STATE_ABBR_Orig'], axis = 1, inplace = True)

	# makeDendrogram(linkage(new_cluster_data_hierarchical, 'single'), range(0, len(new_cluster_data_hierarchical["AVG_REL_EST_TOTAL_PER_CAPITA"])))

if __name__ == '__main__':
	main()




