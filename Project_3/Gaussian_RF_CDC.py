# Python code for analyzing USCS data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import csv
from pandas.plotting import scatter_matrix
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

#setting display options
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)


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
  print("\n" + "Median for Population: " + str(myData['Population'].median()) + "\n")

  return myData

# find outliers in data by column using z score, but keep for analysis
def find_outliers(myData):
# pprint(myData)
  # not necessary to work with all attributes in the
  # data frame, these three are the most relevant
  cols = ['AgeAdjustedRate', 'CaseCount', 'Population']

  # using scipy package to get z score
  z_scores = myData[cols].apply(zscore)
  print("Z scores for AgeAdjustedRate, CaseCount, and Population" + "\n")
  #pprint(z_scores)

  # tally of outliers in each column, we use -2.5 and 2.5 z score
  # as the lower and upper bounds for determining if outlier or not
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

def normalize_data(myData):
  non_numerical_data = ['Area']
  myData[non_numerical_data] = myData[non_numerical_data].apply(LabelEncoder().fit_transform)

  # normalizing data, this operation is only done on continuous attributes
  attributes_to_normalize = ['AgeAdjustedRate', 'lci', 'uci', 'CaseCount', 'Population']
  myData[attributes_to_normalize] = myData[attributes_to_normalize].apply(lambda x:(x-x.mean())/(x.std()))
  return myData

# normalizing merged dataset for predictive models
def normalize_merged_data(myData):
  categorical_vars = ['STATE_ABBR', 'CHEM_RATE_LEVEL', 'CANCER_RATE_LEVEL']
  #attributes_to_normalize = [] #['AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE']  #'YEAR'
  pprint(myData)
  myData[categorical_vars] = myData[categorical_vars].apply(LabelEncoder().fit_transform)
  #myData[attributes_to_normalize] = myData[attributes_to_normalize].apply(lambda x:(x-x.mean())/(x.std()))
  pprint(myData)
  return myData

# call on merged CDC and EPA dataframe
# using z-score, determine if rate of cancer and toxin release in a state
# is very low (below -2.5 inclusive), low (-2.5 to -1 inclusive), 
# medium (-1 to 1 inclusive), high (1 to 2.5 inclusive), and very high (above 2.5)
# also creates 2 new columns for the categorizing the levels
def get_category_of_cancer_rate(myData):
  cols = ['AVG_REL_EST_TOTAL_PER_CAPITA', 'AGE_ADJUSTED_CANCER_RATE']

  # drops rows that don't have cancer data
  myData = myData[myData['AGE_ADJUSTED_CANCER_RATE'].notnull()]

  z_scores = myData[cols].apply(zscore)

  chem_list = z_scores['AVG_REL_EST_TOTAL_PER_CAPITA'].tolist()
  cancer_list = z_scores['AGE_ADJUSTED_CANCER_RATE'].tolist()

  myData['Z_SCORE_AVG_REL_EST_TOTAL'] = chem_list
  myData['Z_SCORE_AGE_ADJUSTED_CANCER_RATE'] = cancer_list

  # this will be either very low, low, medium, etc
  cancer_rate_level = []
  chem_rate_level = []
  x = " "

  for i in cancer_list:
    if (i <= -2.5):
      cancer_rate_level.append("very low")
    elif ((i > -2.5) & (i <= -1)):
      cancer_rate_level.append("low")
    elif ((i > -1) & (i <= 1)):
      cancer_rate_level.append("medium")
    elif ((i > 1) & (i <= 2.5)):
      cancer_rate_level.append("high")
    elif (i > 2.5):
      cancer_rate_level.append("very high")
    else:
      cancer_rate_level.append("test")

  for i in chem_list:
    if (i <= -2.5):
      chem_rate_level.append("very low")
    elif ((i > -2.5) & (i <= -1)):
      chem_rate_level.append("low")
    elif ((i > -1) & (i <= 1)):
      chem_rate_level.append("medium")
    elif ((i > 1) & (i <= 2.5)):
      chem_rate_level.append("high")
    elif (i > 2.5):
      chem_rate_level.append("very high")
    else:
      chem_rate_level.append("test")

  myData['CHEM_RATE_LEVEL'] = chem_rate_level
  myData['CANCER_RATE_LEVEL'] = cancer_rate_level

  print("Z scores for AGE_ADJUSTED_CANCER_RATE" + "\n")
  #pprint(z_scores)
  #print("z score binned data")
  #pprint(myData)

  return myData

# makes naive bayes and random forest classifiers
# analyzes merged cdc and epa data
def predictive_models(myData, numAttributes):
  scoring = 'accuracy'
  pprint(myData.columns)
  valueArray = myData.values
  X = valueArray[:, 0:numAttributes]
  Y = valueArray[:, numAttributes]
  test_size = 0.25
  seed = 7
  X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

  # gaussian naive bayes classifier
  gaussian_nb = GaussianNB().fit(X_train, Y_train)
  random_forest = RandomForestClassifier(n_estimators = 50, random_state=12).fit(X_train, Y_train)

  models = []

  models.append(('random_forest', RandomForestClassifier()))
  models.append(('gaussian_nb', GaussianNB()))

  results = []
  names = []

  for name, model in models:
    model.fit(X_train, Y_train)
    x_predictions =  model.predict(X_validate) # etc
    pprint(x_predictions)
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
  myData.hist()
  plt.tight_layout()
  plt.show()
  plt.clf()

  #goes through each column and makes a histogram for it. if no histogram possible, makes a line graph
  for column in myData.columns:
    print(column)
    try:
      myData[column].hist(bins=(myData[column].max()+1 - myData[column].min()))
      plt.xticks(np.arange(myData[column].min(), myData[column].max(), 1))
      plt.show()
    except:
      plt.clf()
      print("histogram not possible")
      mean_list = []

      color_counter = 0
      colors = ["black"]

      for each in myData['Area'].unique():   #for each state, get the average value of the column for year year and plot that on the y-axis
        for yr in range(myData['Year'].min(), myData['Year'].max(), 1):
          mean_list.append((myData.loc[myData['Area'] == each].loc[myData['Year'] == yr])[column].mean())
        
        lineGraphByRegion(list(range(myData['Year'].min(), myData['Year'].max())), mean_list, colors[color_counter], each, column, "Mean Values for Each State Over Time", each)
        mean_list = []

      plt.show()
      plt.clf()
  
  # Scatterplots to look at 2 variables at once
  # scatter plot matrix

  scatter_matrix(myData)
  plt.savefig("USCS_plots.png")
  plt.show()

# writes out dataframe
def write_to_file(myData, file_name):
  myData.to_csv(file_name)

# main
def main():
  merged_df = pd.read_csv('merged_data2.csv', sep=',', encoding='latin1')

  #only keep year, state, chemicals and cancer adta
  merged_df2 = merged_df.loc[:, merged_df.columns.intersection(['YEAR', 'STATE_ABBR', 'AVG_REL_EST_TOTAL_PER_CAPITA', 'AGE_ADJUSTED_CANCER_RATE'])]

  #drop null values
  merged_df2.dropna(inplace = True)
  
  #get the cancer rate bins based on z-score
  new_df = get_category_of_cancer_rate(merged_df2)

  #print(new_df)

  #normalize the categorical columns via lebl encoding
  new_df = normalize_merged_data(new_df)

  #drop all cancer-related colums so that it can be predicted (except for the rate level)
  new_df.drop(['AGE_ADJUSTED_CANCER_RATE', 'Z_SCORE_AVG_REL_EST_TOTAL', 'Z_SCORE_AGE_ADJUSTED_CANCER_RATE'], axis = 1, inplace = True)

  print("new stuff")
  print(new_df.columns)

  #run random forrest and naive bayes predictive models
  predictive_models(new_df, len(new_df.columns)-1)
  print("end main")

if __name__ == '__main__':
    main()