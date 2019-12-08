#separates the cdc data by the column for the state and returns the dataframe with the region column based on state timezone. Abbreviates state name if necessary
def separateByRegion(dataframe, state_label, has_abbrev):

	if (has_abbrev == False):

		dataframe[state_label] = dataframe[state_label].apply(lambda x: abbreviate(x))

		dataframe['region'] = dataframe[state_label].apply(lambda x: categorization(x))
	        
		return dataframe

	else:
		dataframe['region'] = dataframe[state_label].apply(lambda x: categorization(x))
		return dataframe

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

def categorize(value):
    #state abbr to region
	if value in ['CT', 'DE', 'FL', 'GA', 'IN', 'KY', 'ME', 'MI', 'MD', 'MA', 'PA', 'OH', 'WV','VA','NC', 'SC', 'NY', 'VT', 'NH', 'RI', 'DC','NJ']:
		return 'EAST'
	elif value in ['ND', 'MN', 'WI', 'SD', 'NE', 'IA', 'IL', 'KS', 'MO', 'TN', 'OK', 'AR', 'MS', 'AL', 'TX', 'LA']:
		return 'CENT'
	elif value in ['MT', 'ID', 'WY', 'UT', 'CO', 'AZ', 'NM']:
		return 'MONT'
	else:
		return 'PACF'


def main():


if __name__ == '__main__':
	main()