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


	new_CDC_data['Area'] = new_CDC_data['Area'].apply(lambda x: abbreviate(x))


	new_CDC_data['region'] = new_CDC_data['Area'].apply(lambda x: categorization(x))
        
	return new_CDC_data

#adds a new column for the age adjusted rate binned by the z-score
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
	
	new_CDC_data['Rate_categ'] = new_CDC_data['AgeAdjustedRate'].apply(lambda x: categorization(x))
        
	return new_CDC_data

#returns the dataframe with values for a particular year
def separate_by_year(CDC_Data, year):
	print("separating data by year")

	CDC_Data_yearly = CDC_Data.loc[CDC_Data['Year'] == year].copy()

	return CDC_Data_yearly

#main driver program to make heat maps for t-test results and cross-correlogram from linear regressions 
def main():
	print("main")

	# Read in data directly into pandas
	cleaned_CDC_Data = pd.read_csv('USCS_CancerTrends_OverTime_ByState.csv' , sep=',', encoding='latin1')

	#normalized_CDC_Data = normalizeCDC(cleaned_CDC_Data, ['AgeAdjustedRate'])   #z score normalization
	normalized_CDC_Data = cleaned_CDC_Data
	normalized_CDC_Data.dropna(inplace = True)

	normalized_CDC_Data = separateByRegion(normalized_CDC_Data)

	binned_CDC_Data = binRate(normalized_CDC_Data)

	pprint(binned_CDC_Data)

	year_start = 1999
	year_end = 2017

	for year in range(year_start, year_end, 1):
		new_binned_CDC_Data = separate_by_year(binned_CDC_Data, year)
		#makeHeatMap_pval(new_binned_CDC_Data, str(year))
		#pprint(binned_CDC_Data)

	mean_list = []

	color_counter = 0;
	colors = ["green", "blue", "red", "black", "purple"]

	for each in binned_CDC_Data['region'].unique():
		for yr in range(year_start, year_end, 1):
			mean_list.append((binned_CDC_Data.loc[binned_CDC_Data['region'] == each].loc[binned_CDC_Data['Year'] == yr])['AgeAdjustedRate'].mean())
			#print(mean_list)
		
		lineGraphByRegion(list(range(year_start, year_end)), mean_list, colors[color_counter], each)

		mean_list = []
		color_counter += 1

	#plt.clf()
	plt.show()
	plt.clf()

	#heat map of linear regression correlations between each regions 
	makeHeatMap_corr(binned_CDC_Data)

#does linear regression for the cdc dataframe and returns the list of R^2 values from each comparison
def doLinearReg(binned_CDC_Data):

	lin_reg_values = []

	for region1 in binned_CDC_Data['region'].unique():
		for region2 in binned_CDC_Data['region'].unique():

			series1 = []
			series2 = []
			for yr in range(1999, 2016, 1):
				series1.append(float((binned_CDC_Data.loc[binned_CDC_Data['region'] == region1].loc[binned_CDC_Data['Year'] == yr])['AgeAdjustedRate'].mean()))   #age adjusted rate for a particular year and region
				series2.append(float((binned_CDC_Data.loc[binned_CDC_Data['region'] == region2].loc[binned_CDC_Data['Year'] == yr])['AgeAdjustedRate'].mean()))

			#scatterPlot(series1, series2, "east", "mont", "correlation", 0)

			linear_model = LinearRegression().fit(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))

			r_sq = linear_model.score(np.array(series1).reshape(-1,1), np.array(series2).reshape(-1,1))
			print('coefficient of determination:' + region1 + " " + region2, r_sq)
			lin_reg_values.append(r_sq)

	return lin_reg_values


#makes a line graph with a line of a given color and region (label for the line)
def lineGraphByRegion(x_data, y_data, color_val, region):

	#plt.xticks(x_data)
	fig = plt.figure(1)
	ax = plt.axes()
	plt.xlabel('Year')
	plt.ylabel("Age Adjusted Cancer Rate (per 100,000 people)")
	ax.plot(x_data, y_data, color = color_val, label = region)
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels)
	#ax.grid('on')

#makes a heatmap of the correlations between regions given the dataframe
def makeHeatMap_corr(binned_CDC_Data):

	lin_reg_matrix = [[],[],[],[],[]]

	lin_reg_values = doLinearReg(binned_CDC_Data)

	#creates a matrix of the correlation values
	counter = 0
	counter2 = 0
	for each in lin_reg_values:
		lin_reg_matrix[counter].append(lin_reg_values[counter2])
		counter2 += 1
		if(counter2 % 5  == 0):   #number of regions
			counter += 1

	#prints matrix
	pprint(lin_reg_matrix)

	#generate heat map and display
	heatMap(" of cancer rate correlation over time between regions", lin_reg_matrix, list(binned_CDC_Data['region'].unique()), list(binned_CDC_Data['region'].unique()))

#makes a heat map of the p values from t-test between age adjusted cancer rate for each region's states
def makeHeatMap_pval(binned_CDC_Data, year_str):

	t_test_matrix = [[],[],[],[],[]]

	p_vals = []

	#run t-tests and append to the list of p values
	for each1 in binned_CDC_Data['region'].unique():
		for each2 in binned_CDC_Data['region'].unique():
			p_vals.append(ttest_ind(binned_CDC_Data.loc[binned_CDC_Data['region'] == each1]['AgeAdjustedRate'], binned_CDC_Data.loc[binned_CDC_Data['region'] == each2]['AgeAdjustedRate']).pvalue)
			print(each1 + str(binned_CDC_Data.loc[binned_CDC_Data['region'] == each1]['AgeAdjustedRate'].mean())+ " "+str(binned_CDC_Data.loc[binned_CDC_Data['region'] == each1]['AgeAdjustedRate'].std()), each2 + str(binned_CDC_Data.loc[binned_CDC_Data['region'] == each2]['AgeAdjustedRate'].mean())+ " "+str(binned_CDC_Data.loc[binned_CDC_Data['region'] == each2]['AgeAdjustedRate'].std()))
			print(ttest_ind(binned_CDC_Data.loc[binned_CDC_Data['region'] == each1]['AgeAdjustedRate'], binned_CDC_Data.loc[binned_CDC_Data['region'] == each2]['AgeAdjustedRate']).pvalue)

	#add p values to the matrix

	counter = 0
	counter2 = 0
	for each in p_vals:

		# if the p value is very very low, then just make it zero for visualization purposese
		p_value = p_vals[counter2]
		if p_value < 0.01:
			p_value = 0.0

		t_test_matrix[counter].append(p_value)
		counter2 += 1
		if(counter2 % 5 == 0):  #number of regions
			counter += 1

	#print matrix
	pprint(t_test_matrix)

	#generate and plot heatmap
	heatMap(" for t-tests of cancer rates between regions for "+year_str, t_test_matrix, list(binned_CDC_Data['region'].unique()), list(binned_CDC_Data['region'].unique()))

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
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	#make the heat map plot
	if not isinstance(heat_map, (list, np.ndarray)):
		heat_map = im.get_array()

	# Normalize the threshold to the images color range.
	threshold = im.norm(heat_map.max())/2.

	# Loop over matrix and create text annotations with the values passed in
	for i in range(len(x_axis)):
	    for j in range(len(y_axis)):
	        text = ax.text(j, i, str(heat_map[i, j])[0:4],
	                       ha="center", va="center", color="b")

	ax.set_title("heatmap"+year_str)

	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	#make the plot fully visible and not cut off on the sides
	ax.set_xticks(np.arange(heat_map.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(heat_map.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
	ax.tick_params(which="minor", bottom=False, left=False)

	#ad the color bar on the side with the heat range
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel("color bar", rotation=-90, va="bottom")

	#display the heatmap
	fig.tight_layout()
	plt.savefig("heatmap"+year_str+'.png')
	#plt.show()
	plt.clf()


if __name__ == '__main__':
	main()
