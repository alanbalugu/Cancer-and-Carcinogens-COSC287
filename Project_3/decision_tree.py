# Import all libraries.
# Adrian Hertel + Alan Balu

import pandas
import numpy as np
from pprint import pprint
from sklearn import preprocessing
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

CATEG_COLS = ['STATE_ABBR']

#function to bin the cancer rate rate based on z-score
def binRate(merged_data):

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
	
	merged_data['AGE_ADJUSTED_CANCER_RATE_NORMALIZED_LABELS'] = merged_data['AGE_ADJUSTED_CANCER_RATE_NORMALIZED'].apply(lambda x: categorization(x))
        
	return merged_data

#function to do the decision tree analysis
def decisionTree():
	######################################################
	# Load the data
	######################################################
	merged_data = pandas.read_csv('merged_data2.csv')

	######################################################
	# transform data 
	######################################################
	# convert categorical columns to numerical form
	encoder = preprocessing.LabelEncoder()
	convertCategoricals(merged_data, encoder)


	######################################################
	# Evaluate algorithms
	######################################################

	# Separate training and final validation data set. First remove class label from data (X). Setup target class (Y)
	# Then make the validation set 20% of the entire set of labeled data (X_test, Y_test)

	# preprocess data to remove null rows with null Y values and replace null values in X set with zeroes.
	merged_data = merged_data.dropna(subset=['AGE_ADJUSTED_CANCER_RATE'])
	merged_data = merged_data.fillna(0)

	rate_series = merged_data['AGE_ADJUSTED_CANCER_RATE']
	z_scores = (rate_series-rate_series.mean())/rate_series.std()
	merged_data['AGE_ADJUSTED_CANCER_RATE_NORMALIZED'] = z_scores
	merged_data = binRate(merged_data)

	merged_data.drop(['AGE_ADJUSTED_CANCER_RATE', 'AGE_ADJUSTED_CANCER_RATE_NORMALIZED'], axis = 1, inplace = True)

	pprint(merged_data.columns)

	valueArray = merged_data.values
	X = valueArray[:, 0:25]
	Y = valueArray[:, 25]
	test_size = 0.20
	seed = 7

	# create boolean column for whether row is an outlier or not
	outlier_list = []
	total_outliers_AVG_EST = 0
	for i in z_scores:
		if (i < -2.5) | (i > 2.5):
			total_outliers_AVG_EST += 1
			outlier_list.append(True)
		else:
			outlier_list.append(False)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	######################################################
	# For the decision tree, see how well it does on the
	# validation test
	######################################################
	decision_tree = DecisionTreeClassifier()
	decision_tree.fit(X_train, Y_train)
	decision_tree_predictions = decision_tree.predict(X_test)

	pprint(decision_tree_predictions)
	print("accuracy_score = ", accuracy_score(Y_test, decision_tree_predictions))
	print(confusion_matrix(Y_test, decision_tree_predictions))
	print(classification_report(Y_test, decision_tree_predictions))


# convers categorical variables to numerial
def convertCategoricals(myData, encoder):
	for col in CATEG_COLS:
		myData[col] = encoder.fit_transform(myData[col])

def main():
	decisionTree()

if __name__ == '__main__':
	main()