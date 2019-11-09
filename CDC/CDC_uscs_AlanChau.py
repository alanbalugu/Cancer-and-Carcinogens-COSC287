# Python code for analyzing USCS data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import csv
from pandas.plotting import scatter_matrix
#from sodapy import Socrata
from pprint import pprint
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pprint import pprint

# calculate mean, mode (for categorical data), median,
# and std for at least 10 attributes, 3 attributes here 
def basic_statistical_analysis(myData):
	myData = myData.dropna(axis=0)
	myData = myData.drop(columns=['CancerType'])
	myData['CaseCount'] = pd.to_numeric(myData['CaseCount'])

	# the pandas describe method calculates mean and std
	# as well as other relevant values like quartile, min,
	# and max
	print(myData.describe())

	# median for the 3 main numeric attributes we're analyzing
	# in the USCS dataset
	print("\n" + "Median for Age Adjusted Rate: " + str(myData['AgeAdjustedRate'].median()))
	print("\n" + "Median for Age Case Count: " + str(myData['CaseCount'].median()))
	print("\n" + "Median for Population: " + str(myData['Population'].median()))

	return myData

# find outliers in data by column using z score, but keep for analysis
def find_outliers(myData):
	# not necessary to work with all attributes in the
	# data frame, these three are the most relevant
	cols = ['AgeAdjustedRate', 'CaseCount', 'Population']
	z_scores = myData[cols].apply(zscore)
	pprint(myData)

	# using scipy package to get z score
	print("Z scores for AgeAdjustedRate, CaseCount, and Population" + "\n")
	pprint(z_scores)

	total_outliers_AAR = 0
	total_outliers_CC = 0
	total_outliers_P = 0

	AAR_list = z_scores['AgeAdjustedRate'].tolist()
	CC_list = z_scores['CaseCount'].tolist()
	P_list = z_scores['Population'].tolist()

	for i in AAR_list:
		if (i < -2.5) | (i > 2.5):
			total_outliers_AAR += 1

	for i in CC_list:
		if (i < -2.5) | (i > 2.5):
			total_outliers_CC += 1

	for i in P_list:
		if (i < -2.5) | (i > 2.5):
			total_outliers_P += 1

	print("total_outliers_AAR: " + str(total_outliers_AAR) + "\n")
	print("total_outliers_CC: " + str(total_outliers_CC) + "\n")
	print("total_outliers_P: " + str(total_outliers_P) + "\n")

# make non-numerical values numerical and normalize data
# remember to drop CancerType column
def normalize_data(myData):
	# replace bad data
#	myData = myData.drop(columns=['CancerType'])
	non_numerical_data = ['Area']
	myData[non_numerical_data] = myData[non_numerical_data].apply(LabelEncoder().fit_transform)

	# normalizing data, this operation is only done on continuous attributes
	attributes_to_normalize = ['AgeAdjustedRate', 'lci', 'uci', 'CaseCount', 'Population']
	myData[attributes_to_normalize] = myData[attributes_to_normalize].apply(lambda x:(x-x.min())/(x.max()-x.min()))

	return myData

# makes random forest and naive bayes classifiers
# need to combine cdc and api before 
def predictive_models(myData, numAttributes):
	# Separate training and final validation data set. First remove class
	# label from data (X). Setup target class (Y)
	# Then make the validation set 20% of the entire
	# set of labeled data (X_validate, Y_validate)

	pprint(myData)
	scoring = 'accuracy'
	valueArray = myData.values
	X = valueArray[:, 0:numAttributes]
	Y = valueArray[:, numAttributes]
	test_size = 0.20
	seed = 7
	X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

	# making random forest classifier
	random_forest = RandomForestClassifier(n_estimators = 50, random_state=12)

	# gaussian naive bayes classifier
	gaussian_nb = GaussianNB()

	models = []

	models.append(('random_forest', RandomForestClassifier()))
	models.append(('gaussian_nb', GaussianNB()))

	# Evaluate each model, add results to a results array,
	# Print the accuracy results (remember these are averages and std
	results = []
	names = []

	for name, model in models:
		model.fit(X_train, Y_train)
		x_predictions =  model.predict(X_validate) # etc
		print(accuracy_score(Y_validate, x_predictions))
		print(confusion_matrix(Y_validate, x_predictions))
		print(classification_report(Y_validate, x_predictions))

#makes a line graph and adds to plot
def lineGraphByRegion(x_data, y_data, color_val, region, y_axis_label, title_label, state):

	#plt.xticks(x_data)
	fig = plt.figure(1)
	ax = plt.axes()
	plt.title(title_label)
	plt.xlabel('X - Axis')
	plt.ylabel(y_axis_label)
	ax.plot(x_data, y_data, color = color_val, label = region)
	handles, labels = ax.get_legend_handles_labels()
	#lgd = ax.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))
	plt.text(x_data[-1], y_data[-1], state, color = "red")
	ax.grid('on')

# plots data	
def plot_data(myData):
	# Box and whisker plots
	myData.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
	plt.tight_layout()
	plt.show()

	# Histogram
	myData.hist(bins=10)
	plt.tight_layout()
	plt.show()

	for column in myData.columns:
		try:
			myData[column].hist(bins=(myData[column].max()+1 - myData[column].min()))
			plt.xticks(np.arange(myData[column].min(), myData[column].max(), 1))
			plt.show()
		except:
			print("histogram not possible")
			mean_list = []

			color_counter = 0;
			colors = ["black"]

			for each in myData['Area'].unique():
				for yr in range(myData['Year'].min(), myData['Year'].max(), 1):
					mean_list.append((myData.loc[myData['Area'] == each].loc[myData['Year'] == yr])[column].mean())
					#print(mean_list)
				
				lineGraphByRegion(list(range(myData['Year'].min(), myData['Year'].max())), mean_list, colors[color_counter], each, column, "Mean Values for Each State Over Time", each)

				mean_list.clear()
				#color_counter += 1

			#plt.clf()
			plt.show()
			#plt.clf()
	
	# Scatterplots to look at 2 variables at once
	# scatter plot matrix
	scatter_matrix(myData)
	plt.show()

# writes out dataframe
def write_to_file(myData, file_name):
	myData.to_csv(file_name)

# main
def main():
	df = pd.read_csv('USCS_CancerTrends_OverTime_ByState.csv' , sep=',', encoding='latin1')
	df = basic_statistical_analysis(df)
	find_outliers(df)
	normalized_df = normalize_data(df)
	pprint(normalized_df)
	plot_data(normalized_df)
	print("end main")

if __name__ == '__main__':
    main()

"""
	myData = myData[cols]
	for i in cols:
		z_score = i + '_z_score'
		myData[z_score] = (myData[i] - myData[i].mean()) / myData[i].std(ddof=0)
"""
