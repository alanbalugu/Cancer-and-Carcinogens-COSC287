#p2 code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import csv
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean
from statistics import median
from statistics import stdev
from math import pi
from pandas.plotting import scatter_matrix
from sodapy import Socrata
from pprint import pprint
from scipy import stats
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import normalize
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

"""
** Avg. Annual Age-adjusted rate is the only rate type concerning
	incidence/mortality, the others have to do with test methods

** For the rates concerning cancer diagnosis method, there
	is race and gender data

After dropping rows with NaN, we have 4 different cancer types remaining,
~7244 rows of data
	Age-adjusted Prevalence, 2012-16
	Average Annual Age-adjusted Rate, 2010-14 range only, CANCER
	Average Annual Crude Rate, 2010-14 range only
	Crude Prevalence, 2012-16

"""

# additional cleaning and reorganizing of cleaned data from project 1
def clean_and_sort(myData):
	myData = myData.drop(['geolocation', 'stratificationcategory1'], axis=1)
	myData = myData.dropna(axis=0)
	myData = myData.sort_values(by=['datavaluetype', 'yearstart', 'locationdesc'])
	return myData

# normalizing continuous data
def normalize(myData, columns):
	new_df = myData

	print("Normalizing...")
	for col in columns:
		maximum = new_df[col].max()
		minimum = new_df[col].min()

		#normalize as z-score 
		new_df[col] = (new_df[col] - new_df[col].mean()) / (new_df[col].std())

	return new_df

# calculate mean, mode (for categorical data), median,
# and std deviation for at least 10 attributes 
# mode is only required for the categorical attribs?
# only do this for the diff cancer rates
def basic_statistical_analysis(myData):
	# df already sorted by 4 rate types
	# create list object from 'datavaluetype' column
	DVT_list = myData['datavaluetype'].tolist()
	DV_list = myData['datavalue'].tolist()

	total_rows_df = len(DVT_list)

	num_age_adj_prev = 0
	num_AA_age_adj_rate = 0
	num_AA_crude_rate = 0
	num_crude_prev = 0

	age_adj_prev = []
	AA_age_adj_rate = []
	AA_crude_rate = []
	crude_prev = []

	for i in DVT_list:
		if i == "Age-adjusted Prevalence":
			num_age_adj_prev += 1
		if i == "Average Annual Age-adjusted Rate":
			num_AA_age_adj_rate += 1
		if i == "Average Annual Crude Rate":
			num_AA_crude_rate += 1
		if i == "Crude Prevalence":
			num_crude_prev += 1

	# substituting shorter variables for convenience of calculating below
	a = num_age_adj_prev
	b = num_AA_age_adj_rate
	c = num_AA_crude_rate
	d = num_crude_prev

	# append values to their respective lists to do some stats analysis
	# on each of the types of cancer rates
	age_adj_prev = DV_list[:a]
	AA_age_adj_rate = DV_list[a + 1:a + b]
	AA_crude_rate = DV_list[a + b + 1: a + b + c]
	crude_prev = DV_list[a + b + c + 1:]

	# find outliers using z score, put lists into a dataframe
	# we want to look at AAAA rate and AA crude rate for cancer
	# incidence / mortality data
	df = pd.DataFrame(list(zip(age_adj_prev, AA_age_adj_rate, AA_crude_rate, crude_prev)), columns = ['AA Prev', 'AA Adj Rate', 'AA Crude Rate', 'Crude Prev'])
	pprint(df)
	write_to_file(df, "rates.csv")

	print(myData.describe())
	print("Count of AA Prev: " + str(num_age_adj_prev) + "\n")
	print("Count of AAAA Rate: " + str(num_AA_age_adj_rate) + "\n")
	print("Count of AA Crude Rate: " + str(num_AA_crude_rate) + "\n")
	print("Count of Crude Prev: " + str(num_crude_prev) + "\n")

	print("Mean, median, and stdev of AA Prev: " + str(mean(age_adj_prev)) + ", " + str(median(age_adj_prev)) + ", " + str(stdev(age_adj_prev)) + "\n")
	print("Mean, median, and stdev of AAAA Rate: " + str(mean(AA_age_adj_rate)) + ", " + str(median(AA_age_adj_rate)) + ", " + str(stdev(AA_age_adj_rate)) + "\n")
	print("Mean, median, and stdev of AA Crude Rate: " + str(mean(AA_crude_rate)) + ", " + str(median(AA_crude_rate)) + ", " + str(stdev(AA_crude_rate)) + "\n")
	print("Mean, median, and stdev of Crude Prevalence: " + str(mean(crude_prev)) + ", " + str(median(crude_prev)) + ", " + str(stdev(crude_prev)) + "\n")

	print("Total rows df: " + str(total_rows_df) + "\n")

# writes out dataframe
def write_to_file(myData, file_name):
	myData.to_csv(file_name)

def main():
	df = pd.read_csv("CDC_API_Clean.csv")
	df = clean_and_sort(df)
	basic_statistical_analysis(df)

if __name__ == '__main__':
    main()

