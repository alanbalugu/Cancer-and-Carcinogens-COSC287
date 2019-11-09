import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from pprint import pprint

def main():
    #read data from cleaned epa files into pandas dataframe
    myDataFrame = pd.read_csv("merged_data.csv", sep=',', encoding='latin1')
    print(myDataFrame[:10])
    #create list of names for columns with numerical data
    numCols = ['AVG_REL_EST_AIR_STACK','AVG_REL_EST_ENERGY_RECOVERY','AVG_REL_EST_RECYCLING',
    'AVG_REL_EST_OTH_DISP','AVG_REL_EST_POTW_NON_METALS','AVG_REL_EST_WATER','AVG_REL_EST_SURF_IMP',
    'AVG_REL_EST_UNINJ_IIV','AVG_REL_EST_AIR_FUG','AVG_REL_EST_LAND_TREA','AVG_REL_EST_WASTE_TREATMENT',
    'AVG_REL_EST_OTH_LANDF','AVG_REL_EST_RCRA_C','AVG_REL_EST_TOTAL_ON_OFFSITE_RELEASE','AVG_REL_EST_UNINJ_I',
    'AVG_REL_EST_SI_5.5.3B','AVG_REL_EST_TOTAL_ONSITE_RELEASE','AVG_REL_EST_SI_5.5.3A','AVG_REL_EST_POTW_RELEASE',
    'AVG_REL_EST_POTW_TREATMENT','AVG_REL_EST_TOTAL','AGE_ADJUSTED_CANCER_RATE']
    #create list of names for columns with categorical data
    catCols = ['STATE_ABBR']

    #add a column that categorizes data by region
    myDataFrame.fillna(0, inplace=True)
    for col in catCols:
        myDataFrame = catToNum(myDataFrame, col)

    myDataFrame = normalizeData(myDataFrame, numCols)
    myDataFrame = binRate(myDataFrame)
    print(myDataFrame[:1000])
    testAccuracy(myDataFrame)

def catToNum(myDataFrame, col):
    numDict = {}
    #find unique values for column
    uniqueList = pd.unique(myDataFrame[col])
    #make a variable to make data numerical, loop through unique values in column
    count = 0
    for item in uniqueList:
        #make sure item is not null
        if item is not np.nan:
            #add the item to dictionary as key that maps to count
            numDict[item] = count
            count+=1

    #replace all of the categorical values with their respective int values
    myDataFrame = myDataFrame.replace(numDict)
    return myDataFrame

def normalizeData(myDataFrame, colList):
	for col in colList:
		#for z-score standardizing
		stddev = myDataFrame[col].std()
		mean = myDataFrame[col].mean()
		#normalize as z-score
		myDataFrame[col] = (myDataFrame[col] - mean) / (stddev)
	return myDataFrame

def binRate(myDataFrame):
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

	myDataFrame['AGE_ADJUSTED_CANCER_RATE_level'] = myDataFrame['AGE_ADJUSTED_CANCER_RATE'].apply(lambda x: categorization(x))
	return myDataFrame

def testAccuracy(myDataFrame):
    valueArray = myDataFrame.values
    #x values should include all columns
    X = valueArray[:, 0:-1]
    #y value is last column
    Y = valueArray[:, -1:]
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

    #test naive bayes and random forest
    models = []
    models.append(('RF', RandomForestClassifier()))

    #random forest
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_validate)
    train_predictions = rf.predict(X_train)

    print()
    print("rf train", accuracy_score(Y_train, train_predictions))
    print("rf validate", accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))

main()
