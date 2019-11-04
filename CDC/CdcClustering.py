
# Libraries
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from sklearn import preprocessing
import pylab as plt
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

#imports for preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)

def doDBScan(CDC_Data, nearness, min_samples):

	new_CDC_data = CDC_Data

	# Compute DBSCAN
	dbscan = DBSCAN(eps=nearness, min_samples=min_samples).fit(new_CDC_data)
	core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
	core_samples_mask[dbscan.core_sample_indices_] = True
	labels = dbscan.labels_

	new_CDC_data["cluster_labels"] = labels

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)

	silhouette_avg = metrics.silhouette_score(new_CDC_data, labels)

	print("Silhouette Coefficient: %0.3f" % silhouette_avg)

	return new_CDC_data, silhouette_avg, n_clusters_

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

def doHierarchical(CDC_Data, k):

	new_CDC_data = CDC_Data

	hierarchical = AgglomerativeClustering(n_clusters = k)
	cluster_labels = hierarchical.fit_predict(new_CDC_data)

	silhouette_avg = silhouette_score(new_CDC_data, cluster_labels)
	print("\nFor n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

	new_CDC_data["cluster_labels"] = cluster_labels

	return new_CDC_data, silhouette_avg


#drop unnecessary columns, one-hot encoding
def preprocessing(CDC_Data, categories, columns_to_drop):
	
	new_CDC_data = CDC_Data
	list_of_dropped = []
	list_of_categ = []

	#return list of dropped columns to save
	for column in columns_to_drop:
		list_of_dropped.append(new_CDC_data[column])


	new_CDC_data.drop(columns_to_drop, axis = 1, inplace = True)

	#print(new_CDC_data.columns)

	#create dummy columns for categorical variables
	for value in categories:
		new_CDC_data = pd.concat([new_CDC_data,pd.get_dummies(new_CDC_data[value])],axis=1)

		# now drop the original categorical columns (only keep dummies), save dropped original to return
		list_of_categ.append(new_CDC_data[value])
		new_CDC_data.drop([value],axis=1, inplace=True)

	#pprint(new_CDC_data)
	#print(new_CDC_data.columns)

	return new_CDC_data, list_of_dropped, list_of_categ


def separateCDCData(CDC_Data):
	print("separating data...")

	CDC_Data_AgeAR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Age")].copy()
	CDC_Data_CrudeR = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Crude")].copy()
	CDC_Data_Number = CDC_Data.loc[CDC_Data['datavaluetype'].str.contains("Number")].copy()

	#pprint(CDC_Data_AgeAR)

	return CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number


def normalize(CDC_Data, columns_to_norm):

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

def separate_by_year(CDC_Data, year):
	print("separating data by year")

	CDC_Data_yearly = CDC_Data.loc[CDC_Data['yearstart'] == year].copy()

	return CDC_Data_yearly


def scatterPlot(X_Data, Y_Data, x_axis, y_axis, title, save):
	
	plt.figure(1)
	plt.scatter(X_Data, Y_Data, s=50)
	plt.title(title)
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)

	if (save == True):
		plt.savefig(y_axis +" by "+ x_axis + '.png')
		plt.clf()
	else:
		plt.show()
		plt.clf()


def cycleClustering(CDC_data, type_clus, range_list, silh_score_list, cluster_vals):

	max_silh_score = 0
	best_cluster = 1

	normalized_CDC_Data = CDC_data

	if (type_clus == "K" or type_clus == "H"):

		for i in range_list:

			if (type_clus == "K"):
				clustered_CDC_data, silh_score = doKMeans(normalized_CDC_Data, i)
			else:
				clustered_CDC_data, silh_score = doHierarchical(normalized_CDC_Data, i)

			cluster_vals.append(i)
			silh_score_list.append(silh_score)

			if (silh_score > max_silh_score):
				max_silh_score = silh_score
				best_cluster = i

		if (type_clus == "K"):
				clustered_CDC_data, silh_score = doKMeans(normalized_CDC_Data, best_cluster)
		else:
				clustered_CDC_data, silh_score = doHierarchical(normalized_CDC_Data, best_cluster)
	
	else:

		best_cluster = 0.0
		best_eps = 0.0

		for i in range_list:
			try:
				clustered_CDC_data, silh_score, num_clusters = doDBScan(normalized_CDC_Data, i, 4)
				cluster_vals.append(i)
				silh_score_list.append(silh_score)

				if (num_clusters > best_cluster):
					best_cluster = num_clusters

				if (silh_score > max_silh_score):
					max_silh_score = silh_score
					best_eps = i
					
			except:
				print("didn't work")

		clustered_CDC_data, silh_score, num_clusters = doDBScan(normalized_CDC_Data, best_eps, 10)	

	print("best clusters: ", best_cluster)
	print("max silh. score: ", max_silh_score)

	return clustered_CDC_data, silh_score_list, cluster_vals


def preprocessing2(CDC_Data, categories, columns_to_drop):

	new_CDC_Data = CDC_Data

	new_CDC_Data.drop(columns_to_drop, axis = 1, inplace = True)

	#label encode all categorical columns
	le = LabelEncoder()
	old_columns_CDC_Data = new_CDC_Data[categories]
	new_CDC_Data[categories] = new_CDC_Data[categories].apply(le.fit_transform)

	return new_CDC_Data, old_columns_CDC_Data


def main():
	# Read in data directly into pandas
	cleaned_CDC_Data = pd.read_csv('CDC_API_Clean.csv' , sep=',', encoding='latin1', index_col = 0)
	#pprint(cleaned_CDC_Data)
	CDC_Data_AgeAR, CDC_Data_CrudeR, CDC_Data_Number = separateCDCData(cleaned_CDC_Data)

	#CDC_Data_Years = separate_by_year(CDC_Data_AgeAR, 2010)
	CDC_Data_Years = CDC_Data_AgeAR.copy()

	cat_columns = ['question']
	drop_columns = ['yearend', 'stratificationcategory1', 'stratification1', 
		'questionid', 'lowconfidencelimit', 'highconfidencelimit', 'geolocation','datavaluetype', 'locationdesc']   #don't drop the yearstart

	#preprocess and encode only this data and drop other columns
	CDC_Data_Years.dropna(inplace = True)

	new_CDC_Data_Years = CDC_Data_Years.copy()
	processed_CDC_Data, list_of_dropped, list_of_categ = preprocessing(new_CDC_Data_Years, cat_columns, drop_columns)

	#pprint(list_of_categ)

	#columns to normalize
	norm_columns = ['datavalue']
	#normalize rate
	normalized_CDC_Data = normalize(processed_CDC_Data, norm_columns)

	#pprint(normalized_CDC_Data)
	#print(normalized_CDC_Data.columns)

	silh_score_list = []
	cluster_vals = []

	clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(normalized_CDC_Data, "K", np.arange(35, 40, 1), silh_score_list, cluster_vals)
	#clustered_CDC_data, silh_score_list, cluster_vals = cycleClustering(normalized_CDC_Data, "D", np.arange(0.3, 10.0, 0.2), silh_score_list, cluster_vals)

	added_CDC_Data = clustered_CDC_data
	for series in list_of_categ:
		added_CDC_Data = pd.concat([added_CDC_Data,  series], axis = 1)

	added_CDC_Data = pd.concat([added_CDC_Data,  list_of_dropped[8]], axis = 1)

	'''
	scatterPlot(cluster_vals, silh_score_list, "cluster size", "silh. score",'silhouette score by cluster size', False)

	scatterPlot(clustered_CDC_data['cluster_labels'], clustered_CDC_data['datavalue'], "cluster labels", "rate", "rate by cluster label", False)

	scatterPlot(added_CDC_Data['question'], added_CDC_Data['datavalue'], "question", "rate", "rate by question", 0)

	#pprint(CDC_Data_Years)

	scatterPlot(added_CDC_Data['locationdesc'], added_CDC_Data['datavalue'], "state", "rate", "rate by state", 0)

	scatterPlot(added_CDC_Data['locationdesc'], added_CDC_Data['cluster_labels'], "state", "cluster label", "rate by question", 0)

	scatterPlot(added_CDC_Data['cluster_labels'], added_CDC_Data['question'], "cluster labels", "question", "rate by question", 0)

	scatterPlot(added_CDC_Data['cluster_labels'], added_CDC_Data['yearstart'], "cluster labels", "question", "rate by question", 0)
	'''
	#------------------------------------------

	new_CDC_Data_Years2 = CDC_Data_Years.copy()

	cat_columns = ['question', 'locationdesc']
	drop_columns = ['yearend', 'yearstart', 'stratificationcategory1', 'stratification1', 
		'questionid', 'lowconfidencelimit', 'highconfidencelimit', 'geolocation','datavaluetype']   #don't drop the yearstart

	#preprocess and encode only this data and drop other columns
	CDC_Data_Years.dropna(inplace = True)

	#label encode the columns that are categorical
	processed_CDC_Data2, list_of_changed = preprocessing2(new_CDC_Data_Years2, cat_columns, drop_columns)

	#columns to normalize
	norm_columns = ['datavalue']
	#normalize rate
	normalized_CDC_Data2 = normalize(processed_CDC_Data2, norm_columns)

	print("PCA on data label encoded WITHOUT clustering")

	doPCA(normalized_CDC_Data2)

	#rename columns into data frame
	list_of_changed.rename(columns={'question' : 'orig_question', 'locationdesc' : 'orig_locat'}, inplace = True)

	pprint(list_of_changed)

	normalized_CDC_Data2 = pd.concat([normalized_CDC_Data2,  list_of_changed], axis = 1)

	question_mapping = {}
	state_mapping = {}

	pprint(normalized_CDC_Data2)
	#create mapping dictionaries
	i = 0
	for unique_vals in normalized_CDC_Data2['orig_locat'].sort_values().unique():
		state_mapping[i] = unique_vals
		i+=1

	i = 0
	for unique_vals in normalized_CDC_Data2['orig_question'].sort_values().unique():
		question_mapping[i] = unique_vals
		i+=1

	print(normalized_CDC_Data2.locationdesc.unique())
	pprint(state_mapping)

	print(normalized_CDC_Data2.question.unique())
	pprint(question_mapping)

	'''
	CDC_Data_Years.to_csv(r'2010_CDC.csv', index = 1, header=True)

	#pprint(clustered_CDC_data)
	clustered_CDC_data.to_csv(r'kmeans_CDC.csv', index = 1, header=True)
	'''


def doPCA(CDC_Data):
	pca = decomposition.PCA()
	pca.fit(CDC_Data)

	print("PCA variances")
	print(CDC_Data.columns)
	print(pca.components_)
	print(pca.explained_variance_ratio_)

if __name__ == '__main__':
	main()


	'''
	plt.figure(1)
	plt.subplot(221)    
	plt.scatter(x_axis, silh_score_list, s=50)
	plt.title(title + division)
	plt.xlabel('cluster size')
	plt.ylabel('silhouette score')
	#plt.show()
	#plt.savefig('silh_scoure_by_cluster_size_' + division + '.png')
	#plt.clf()

	plt.subplot(222)   
	plt.scatter(clustered_CDC_data['cluster_labels'], clustered_CDC_data['datavalue'], s=50)
	plt.title("rate by clusters")
	plt.xlabel('cluster')
	plt.ylabel(division)
	#plt.show()
	#plt.clf()

	plt.subplot(223)   
	plt.scatter(CDC_Data_Years['question'], CDC_Data_Years['datavalue'], s=50)
	plt.title("rate by question")
	plt.xlabel('questions')
	plt.ylabel(division)
	#plt.show()
	#plt.clf()

	plt.subplot(224)   
	plt.scatter(CDC_Data_Years['locationdesc'], CDC_Data_Years['datavalue'], s=50)
	plt.title("rate by state")
	plt.xlabel('state')
	plt.ylabel(division)
	plt.show()
	plt.clf()


	added_CDC_Data = clustered_CDC_data
	for series in list_of_categ:
		added_CDC_Data = pd.concat([added_CDC_Data,  series], axis = 1)
	

	plt.figure(1)
	plt.subplot(121)
	plt.scatter(added_CDC_Data['locationdesc'], added_CDC_Data['cluster_labels'], s=50)
	plt.title("cluster label by state")
	plt.xlabel('state')
	plt.ylabel("cluster label")
	#plt.show()
	#plt.clf() 

	plt.subplot(122)
	added_CDC_Data.sort_values(by=['cluster_labels'], inplace = True)
	plt.scatter(added_CDC_Data['cluster_labels'], added_CDC_Data['question'], s=50)
	plt.title("question by cluster label")
	plt.xlabel('cluster label')
	plt.ylabel("cluster label")
	plt.show()
	plt.clf()

'''

	'''
	for i in np.arange(35, 50, 1):
		#clustered_CDC_data, silh_score = doKMeans(normalized_CDC_Data, i)
		clustered_CDC_data, silh_score = doHierarchical(normalized_CDC_Data, i)
		cluster_vals.append(i)
		silh_score_list.append(silh_score)

		if (silh_score > max_silh_score):
			max_silh_score = silh_score
			best_cluster = i

	clustered_CDC_data, silh_score = doHierarchical(normalized_CDC_Data, best_cluster)
	

	cluster_range = np.arange(0.3, 10.0, 0.2)

	best_cluster = 0.0

	best_eps = 0.0

	for i in cluster_range:
		try:
			clustered_CDC_data, silh_score, num_clusters = doDBScan(normalized_CDC_Data, i, 4)
			cluster_vals.append(i)
			silh_score_list.append(silh_score)

			if (num_clusters > best_cluster):
				best_cluster = num_clusters

			if (silh_score > max_silh_score):
				max_silh_score = silh_score
				best_eps = i
				
		except:
			print("didn't work")

	#clustered_CDC_data, silh_score, num_clusters = doDBScan(normalized_CDC_Data, best_cluster, 10)	
	clustered_CDC_data, silh_score, num_clusters = doDBScan(normalized_CDC_Data, best_eps, 4)

	'''