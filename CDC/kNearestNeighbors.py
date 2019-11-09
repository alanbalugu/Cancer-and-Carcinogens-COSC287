#Alan Balu

#import statements
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint

from sklearn import decomposition

#imports for preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

#import for splitting test and training set
from sklearn.model_selection import train_test_split

#Import for summary of classification model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from CDC_USCS_clustering import separateByRegion    #(CDC_Data, state_label, is_abbrev):    dataframe
from CdcClustering import CDCpreprocessing2    #(CDC_Data, categories, columns_to_drop):    new_CDC_Data, old_columns_CDC_Data

from CdcClustering import normalizeCDC   #(CDC_Data, columns_to_norm):   CDC_data_norm


#setting display options
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)


#function (modified from Chau's code) to do the K Nearest Neighbors Classification
def predictive_KNN_model(myData):
	# Separate training and final validation data set. First remove class
	# label from data (X). Setup target class (Y)
	# Then make the validation set 20% of the entire
	# set of labeled data (X_validate, Y_validate)

	pprint(myData)
	scoring = 'accuracy'
	valueArray = myData.values
	X = valueArray[:, 0:-1]
	Y = valueArray[:, -1:]
	test_size = 0.20
	seed = 7
	X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)


	# Make predictions on validation dataset
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)

	knn_predictions = knn.predict(X_validate)
	knn_accuracy = knn.predict(X_train)
	print("test set accuracy: ", accuracy_score(Y_train, knn_accuracy))

	#print the relevant scores for the classifier
	print("KNN report: ")
	print(accuracy_score(Y_validate, knn_predictions))
	print(confusion_matrix(Y_validate, knn_predictions))
	print(classification_report(Y_validate, knn_predictions))

	doROCCurve(knn, X_validate, Y_validate)

#plots the ROC curve for the binary classification task
def doROCCurve(knn, X_validate, Y_validate):

	# Compute ROC curve and ROC area for each class
	y_scores = knn.predict_proba(X_validate)
	fpr, tpr, threshold = roc_curve(Y_validate, y_scores[:, 1])
	roc_auc = auc(fpr, tpr)

	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.title('ROC Curve of kNN')
	plt.show()


'''
#bins the cancer rate (after z-sore normalization) and adds a new column for the binned label
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
	
	new_CDC_data['AGE_ADJUSTED_CANCER_RATE_level'] = new_CDC_data['AGE_ADJUSTED_CANCER_RATE'].apply(lambda x: categorization(x))
    
	return new_CDC_data
'''

#bin the cancer rate into a binary class label for the classification task
def binRate(CDC_Data):
	new_CDC_data = CDC_Data

	def categorization(value):
		if value > 0.0:
			return 'above_avg'
		else:
			return  'below_avg'
	
	new_CDC_data['AGE_ADJUSTED_CANCER_RATE_level'] = new_CDC_data['AGE_ADJUSTED_CANCER_RATE'].apply(lambda x: categorization(x))
    
	return new_CDC_data

#driver to do the KNN classification with a binary class label
def main():
	print("main")

	merged_data = pd.read_csv('merged_data.csv' , sep=',', encoding='latin1')

	normalized_data = normalizeCDC(merged_data, ['AGE_ADJUSTED_CANCER_RATE'])   #normalize age adjusted rate column z-score

	region_data = separateByRegion(normalized_data, 'STATE_ABBR', True)

	binnedRate_data = binRate(region_data)  #add column for rate bin level

	print(binnedRate_data.columns)

	binnedRate_data = binnedRate_data.dropna(subset=['AGE_ADJUSTED_CANCER_RATE'])

	#pprint(region_CDC_data)
	categ_columns = ['STATE_ABBR', 'AGE_ADJUSTED_CANCER_RATE_level' ,'region']
	drop_columns = [] #remove columns that are unnecessary for clustering question

	processed_CDC_data, old_columns_CDC_Data = CDCpreprocessing2(binnedRate_data, categ_columns, drop_columns)   #(CDC_Data, categories, columns_to_drop):    new_CDC_data, (series) list_of_dropped, list_of_categ
	processed_CDC_data.fillna(0, inplace = True) 

	predictive_KNN_model(processed_CDC_data)


if __name__ == '__main__':
	main()


