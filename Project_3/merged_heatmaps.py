#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import colorbar

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

#import from other files
from CdcClustering import normalizeCDC   #(CDC_Data, columns_to_norm):   CDC_data_norm
from CdcClustering import scatterPlot    #(X_Data, Y_Data, x_axis, y_axis, title, save):
from CdcClustering import cycleClustering    #(CDC_data, type_clus, range_list, silh_score_list, cluster_vals):   clustered_CDC_data, silh_score_list, cluster_vals
from AssociationRuleMining import categorization
from AssociationRuleMining import abbreviate


#separates the cdc data by the region after abbreviating the state to two letter
def separateByRegion(CDC_Data):

	new_CDC_data = CDC_Data

	new_CDC_data['REGION'] = new_CDC_data['STATE_ABBR'].apply(lambda x: categorization(x))
        
	return new_CDC_data

#returns the dataframe with values for a particular year
def separate_by_year(CDC_Data, year):
	print("separating data by year")

	CDC_Data_yearly = CDC_Data.loc[CDC_Data['YEAR'] == year].copy()

	return CDC_Data_yearly

def doLinearRegr(series1, series2):
	try:
		linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))

		r_sq = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
	except:
		print("error")

	return r_sq

# #makes a line graph with a line of a given color and region (label for the line)
# def lineGraphByRegion(x_data, y_data, color_val, region):

# 	#plt.xticks(x_data)
# 	fig = plt.figure(1)
# 	ax = plt.axes()
# 	plt.xlabel('Year')
# 	plt.ylabel("rate of cancer")
# 	ax.plot(x_data, y_data, color = color_val, label = region)
# 	handles, labels = ax.get_legend_handles_labels()
# 	lgd = ax.legend(handles, labels)
# 	ax.grid('on')

#compute the correlation between states and return the matrix of those correlations for heatmap
def stateWideCorrel(merged_data, data_label1, data_label2):
	print("do correlations")

	lin_reg_matrix = [ [] for i in range(51) ]

	lin_reg_values = []

	#compare each state to every other states
	for state1 in merged_data['STATE_ABBR'].unique():
		for state2 in merged_data['STATE_ABBR'].unique():

			series1 = merged_data.loc[merged_data['STATE_ABBR'] == state1][data_label1]
			series2 = merged_data.loc[merged_data['STATE_ABBR'] == state2][data_label2]


			if (series1.size != series2.size):
				if (series1.size < series2.size):
					series2 = series2[(series2.size - series1.size):]
				else:
					series1 = series1[(series1.size - series2.size):]

			try:
				#perform linear regression
				linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))

				r_sq = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
				print(state1, state2, r_sq)
				#print('coefficient of determination:' + state1 + " " + state2, r_sq)
				lin_reg_values.append(r_sq)
			except:
				lin_reg_values.append(0)

	print("length of linear matrix: ", len(lin_reg_values))
	print("number of states: ", len(merged_data['STATE_ABBR'].unique()))

	#creates a matrix of the correlation values
	counter = 0
	counter2 = 0
	for each in lin_reg_values:
		#print(counter, " " , counter2)
		lin_reg_matrix[counter].append(lin_reg_values[counter2])
		counter2 += 1
		if((counter2 % len(merged_data['STATE_ABBR'].unique())) == 0):   
			counter += 1

	#prints matrix
	#pprint(lin_reg_matrix)

	return lin_reg_matrix

#make a heat map given the title, matrix, axes labels, and color scheme
def heatMap(year_str, matrix, x_axis, y_axis, cbar_kw={}):

	heat_map = np.array(matrix)

	fig = plt.figure(1)
	ax = plt.axes()
	im = ax.imshow(heat_map, cmap = 'Spectral')

	#set axes ticks
	ax.set_xticks(np.arange(len(x_axis)))
	ax.set_yticks(np.arange(len(y_axis)))

	#set axes labels
	ax.set_xticklabels(x_axis)
	ax.set_yticklabels(y_axis)

	#Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
	         rotation_mode="anchor")

	#make the heat map plot
	if not isinstance(heat_map, (list, np.ndarray)):
		heat_map = im.get_array()

	# Normalize the threshold to the images color range.
	threshold = im.norm(heat_map.max())/2.

	ax.set_title("heatmap"+year_str)

	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	#make the plot fully visible and not cut off on the sides
	ax.set_xticks(np.arange(heat_map.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(heat_map.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
	ax.tick_params(which="both", bottom=False, left=False, labelsize = 7)

	#ad the color bar on the side with the heat range
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel("color bar", rotation=-90, va="bottom")

	#display the heatmap
	fig.tight_layout()
	plt.show()
	#plt.savefig("heatmap"+year_str+'.png')
	plt.clf()		


#main driver program to make heat maps for t-test results and cross-correlogram from linear regressions 
def main():
	print("main")

	# Read in data directly into pandas
	merged_data = pd.read_csv('merged_data2.csv' , sep=',', encoding='latin1')

	new_data = merged_data.copy()
	region_data = separateByRegion(new_data)

	#pprint(region_data)

	year_start = 1999
	year_end = 2017

	region_data = region_data.loc[:, region_data.columns.intersection(['YEAR', 'REGION', 'STATE_ABBR', 'AVG_REL_EST_TOTAL_PER_CAPITA', 'AGE_ADJUSTED_CANCER_RATE'])]
	region_data.dropna(inplace = True)
	#region_data.sort_values(by=['REGION'], inplace = True)

	for year in region_data["YEAR"].unique():
		new_data = region_data.loc[region_data['YEAR'] == year]
		#new_data = region_data

		series2 = new_data['AGE_ADJUSTED_CANCER_RATE']
		series1 = new_data['AVG_REL_EST_TOTAL_PER_CAPITA']   #.apply(np.log10)

		#scatterPlot(series1, series2, "total chemical", "cancer rate", "scatter " + str(year), False)
		plt.clf()

		r_squar = doLinearRegr(series1, series2)
		#print('coefficient of determination:' + str(r_squar))

	region_data.sort_values(by=['STATE_ABBR', 'YEAR'], inplace = True)

	pprint(region_data)
	
	#make a heatmap of linear regression between states for cancer rate
	regressionMatrix = stateWideCorrel(region_data, 'AGE_ADJUSTED_CANCER_RATE', 'AGE_ADJUSTED_CANCER_RATE')   #order based on order in data file

	heatMap(" of cancer correlations between states", regressionMatrix, list(region_data['STATE_ABBR'].unique()), list(region_data['STATE_ABBR'].unique()))


if __name__ == '__main__':
	main()
