#Alan Balu

# Libraries to import
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn import metrics

from sklearn import preprocessing
import pylab as plt
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

#imports for preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder


#normalizes CDC (or any other data) using a z-score method (assumes normal distr) by another category. Returns dataframe with normalized column
def normalizeCDC_byQuestion(CDC_data, feature_to_sort_by, feature_to_clean):

	new_CDC_data = CDC_data.copy()

	for each in new_CDC_data[feature_to_sort_by].unique():
		#can be used for min-max scaling alternatively
		#max_value = new_CDC_data.loc[cleaned_CDC_Data[feature_to_sort_by] == each][feature_to_clean].max()
		#min_value = new_CDC_data.loc[cleaned_CDC_Data[feature_to_sort_by] == each][feature_to_clean].min()

		#for z-score standardizing
		stdv_value = new_CDC_data.loc[new_CDC_data[feature_to_sort_by] == each][feature_to_clean].std()
		mean_value = new_CDC_data.loc[new_CDC_data[feature_to_sort_by] == each][feature_to_clean].mean()

		new_CDC_data.loc[new_CDC_data[feature_to_sort_by] == each, feature_to_clean] = (new_CDC_data.loc[new_CDC_data[feature_to_sort_by] == each, feature_to_clean] - mean_value)/stdv_value

	return new_CDC_data

#runs DBScan with a certain value of epsilon and min_samples. Returns dataframe with cluster labels, silhouette score, and number of clusters found
def doDBScan(CDC_Data, nearness, min_samples):

	new_CDC_data = CDC_Data.copy()

	#Creates a clustering model and fits it to the data
	dbscan = DBSCAN(eps=nearness, min_samples=min_samples).fit(new_CDC_data)
	
	#core samples mask
	core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
	core_samples_mask[dbscan.core_sample_indices_] = True
	labels = dbscan.labels_

	#print(dbscan.labels_.tolist())

	#convert cluster labels to a string list
	labels_list = []
	for each in labels.tolist():
		labels_list.append(str(each))

	#add the cluster labels to the dataframe
	new_CDC_data['cluster_labels'] = labels_list


	# Number of clusters in labels, ignoring noise points (-1 cluster) if present
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	#print the relevant values for the DCScan clustering
	print("eps: ", nearness)
	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)

	silhouette_avg = metrics.silhouette_score(new_CDC_data, labels)

	print("Silhouette Coefficient: %0.3f" % silhouette_avg)

	return new_CDC_data, silhouette_avg, n_clusters_


#runs Kmeans clustering with the dataframe and a given values of k. Returns the dataframe with cluster labels and the silhouette score
def doKMeans(CDC_Data, k):

	new_CDC_data = CDC_Data

	#do the actual k-means analysis
	kmeans = KMeans(n_clusters=k)
	cluster_labels = kmeans.fit_predict(new_CDC_data)

	# display clustering accuracy
	silhouette_avg = silhouette_score(new_CDC_data, cluster_labels)
	print("\nFor n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

	new_CDC_data["cluster_labels"] = cluster_labels

	return new_CDC_data, silhouette_avg

#runs heirarchical agglomerative clustering with the dataframe and a given values of k. Returns the dataframe with cluster labels and the silhouette score.
def doHierarchical(CDC_Data, k):

	new_CDC_data = CDC_Data

	#do hierarchical clustering and fit to data
	hierarchical = AgglomerativeClustering(n_clusters = k)
	cluster_labels = hierarchical.fit_predict(new_CDC_data)

	silhouette_avg = silhouette_score(new_CDC_data, cluster_labels)
	print("\nFor n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

	new_CDC_data["cluster_labels"] = cluster_labels

	return new_CDC_data, silhouette_avg


#drop unnecessary columns and performs one-hot encoding. Returns dataframe that is encoded, list of dropped columns (and column data), list of encoded columns (and data)
def CDCpreprocessing(CDC_Data, categories, columns_to_drop):
	
	new_CDC_data = CDC_Data
	list_of_dropped = []
	list_of_categ = []

	#return list of dropped columns to save
	for column in columns_to_drop:
		list_of_dropped.append(new_CDC_data[column])


	new_CDC_data.drop(columns_to_drop, axis = 1, inplace = True)

	#create dummy columns for categorical variables
	for value in categories:
		new_CDC_data = pd.concat([new_CDC_data,pd.get_dummies(new_CDC_data[value])],axis=1)

		# now drop the original categorical columns (only keep dummies), save dropped original to return
		list_of_categ.append(new_CDC_data[value])
		new_CDC_data.drop([value],axis=1, inplace=True)

	return new_CDC_data, list_of_dropped, list_of_categ

#separates data by the type of cancer rate and returns dataframes with rows that have those types of canter rates
def separateCDCData(CDC_Data):
	print("separating data...")

	CDC_Data_AgeAR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Age")].copy()
	CDC_Data_CrudeR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Crude")].copy()
	CDC_Data_Number = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Number")].copy()

	return CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number

#normalizes a list of columns of a data frame. Returns the dataframe normalized.
def normalizeCDC(CDC_Data, columns_to_norm):

	CDC_data_norm = CDC_Data

	print("normalizing...")
	for feature_name in columns_to_norm:
		max_value = CDC_data_norm[feature_name].max()
		min_value = CDC_data_norm[feature_name].min()

		#for z-score standardizing
		stdv_value = CDC_data_norm[feature_name].std()
		mean_value = CDC_data_norm[feature_name].mean()

		#normalize by range
		#CDC_data_norm[feature_name] = (CDC_data_norm[feature_name] - min_value) / (max_value - min_value)

		#normalize as z-score 
		CDC_data_norm[feature_name] = (CDC_data_norm[feature_name] - mean_value) / (stdv_value)

	return CDC_data_norm

#separates the CDC dataframe by year
def separate_by_year(CDC_Data, year):
	print("separating data by year")

	CDC_Data_yearly = CDC_Data.loc[CDC_Data['yearstart'] == year].copy()

	return CDC_Data_yearly


#creates a scatter plot and color codes the values by cluster labels if that parameter is passed in
def scatterPlot2(X_Data, Y_Data, x_axis, y_axis, title, save, clusterLabels = None):
	
	plt.figure(1)

	if (clusterLabels != None):
		plt.scatter(X_Data, Y_Data, s=20, cmap = 'rainbow', c=clusterLabels)
	else:
		plt.scatter(X_Data, Y_Data, s=20)

	plt.title(title)
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)

	if (save == True):
		plt.savefig(y_axis +" by "+ x_axis + '.png')
		plt.clf()
	else:
		plt.show()
		plt.clf()

#creates a scatter plot. 
def scatterPlot(X_Data, Y_Data, x_axis, y_axis, title, save):
	
	plt.figure(1)
	plt.scatter(X_Data, Y_Data, s=20)
	plt.title(title)
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)

	if (save == True):
		plt.savefig(y_axis +" by "+ x_axis + '.png')
		plt.clf()
	else:
		plt.show()
		plt.clf()

#runs clustering for a dataframe through a range of values (k for kmeans, eps for dbscan). Returns dataframe with best clustering cluster labels (max silhouette score) 
def cycleClustering(CDC_data, type_clus, range_list, silh_score_list, cluster_vals):

	max_silh_score = 0
	best_cluster = 1
	min_samples = 4

	normalized_CDC_Data = CDC_data

	#for K means or hierarchical clustering
	if (type_clus == "K" or type_clus == "H"):

		#runs clustering on all values of k in range
		for i in range_list:

			if (type_clus == "K"):
				clustered_CDC_data, silh_score = doKMeans(normalized_CDC_Data, i)
			else:
				clustered_CDC_data, silh_score = doHierarchical(normalized_CDC_Data, i)

			cluster_vals.append(i)
			silh_score_list.append(silh_score)

			#save the k values for the 'best' clustering
			if (silh_score > max_silh_score):
				max_silh_score = silh_score
				best_cluster = i

		#do a final clustering with the best parameter
		if (type_clus == "K"):
				clustered_CDC_data, silh_score = doKMeans(normalized_CDC_Data, best_cluster)
		else:
				clustered_CDC_data, silh_score = doHierarchical(normalized_CDC_Data, best_cluster)
	
	#for dbscan clustering
	else:

		best_cluster = 0.0
		best_eps = 0.1

		#runs dbscan on all epsioln values in the range
		for i in range_list:
			try:
				clustered_CDC_data, silh_score, num_clusters = doDBScan(normalized_CDC_Data, i, min_samples)
				cluster_vals.append(i)
				silh_score_list.append(silh_score)

				if (num_clusters > best_cluster):
					best_cluster = num_clusters
				
				#save the k values for the 'best' clustering
				if (silh_score > max_silh_score):
					max_silh_score = silh_score
					best_eps = i

				clustered_CDC_data = pd.DataFrame()
					
			except:
				print("didn't work")

		#do a final clustering with the best parameter
		clustered_CDC_data, silh_score, num_clusters = doDBScan(CDC_data, best_eps, min_samples)


	print("best clusters: ", best_cluster)
	print("max silh. score: ", max_silh_score)

	return clustered_CDC_data, silh_score_list, cluster_vals


#preprocessed the CDC data using label encoding. Returns the dataframe with the encoded columns and list of the columns (and data) that was dropped
def CDCpreprocessing2(CDC_Data, categories, columns_to_drop):

	new_CDC_Data = CDC_Data

	new_CDC_Data.drop(columns_to_drop, axis = 1, inplace = True)

	#label encode all categorical columns
	le = LabelEncoder()
	old_columns_CDC_Data = new_CDC_Data[categories]
	new_CDC_Data[categories] = new_CDC_Data[categories].apply(le.fit_transform)

	return new_CDC_Data, old_columns_CDC_Data

def main():
	print('Cdc clustering py main....')
	print('doesnt do much just has a lot of functions for use in other files')

def doPCA(CDC_Data):
	pca = decomposition.PCA()
	pca.fit(CDC_Data)

	print("PCA variances")
	print(CDC_Data.columns)
	print(pca.components_)
	print(pca.explained_variance_ratio_)

if __name__ == '__main__':
	main()