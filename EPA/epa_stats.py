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
    myDataFrame1 = pd.read_csv("epa_data_state_releases_cleaned.csv", sep=',', encoding='latin1')
    myDataFrame2 = pd.read_csv("epa_data_state_chems_and_releases_cleaned.csv", sep=',', encoding='latin1')
    myDataFrame3 = pd.read_csv("epa_state_year_frame.csv", sep=',', encoding='latin1')

    #create list of names for columns with numerical data
    numCols = ['SUM_REL_EST','AVG_REL_EST','MIN_REL_EST','MAX_REL_EST','STD_REL_EST','VAR_REL_EST']
    #create list of names for columns with categorical data
    catCols1 = ['STATE_ABBR', 'CATEGORY']
    catCols2 = ['STATE_ABBR', 'CATEGORY', 'CHEM_NAME', 'LIST_3350', 'CARCINOGEN','CLEAN_AIR', 'SRS_ID']

    #add a column that categorizes data by region
    myDataFrame1 = addRegionCat(myDataFrame1)
    myDataFrame2 = addRegionCat(myDataFrame2)
    print(myDataFrame1[:10])

    #run stats tests on the data
    stats1 = runStats(myDataFrame1, numCols)
    stats2 = runStats(myDataFrame2, numCols)

    #anova test that looks at each numerical value across regions
    print('ANOVA:')
    print()
    for col in numCols:
        print('  ' + col + ': ')
        fvalue, pvalue = anovaTest(myDataFrame1, col)
        print('    f = ' + str(fvalue) + ', p = ' + str(pvalue))
    print()

    #make scatterplots and find linear regressions for the numerical data
    linCoefs = {}
    for colA in numCols:
        for colB in numCols:
            if colA is not colB:
                linCoef = makeScatterplot(myDataFrame1, colA, colB)
                linCoefs[(colA,colB)] = linCoef
    print(linCoefs)

    #make histograms for each column
    for col in numCols:
        newDataFrame = myDataFrame1
        newDataFrame = normalizeByFeature(newDataFrame, 'CATEGORY', col)
        newDataFrame = newDataFrame.loc[newDataFrame[col] < 2.5]
        newDataFrame = newDataFrame.loc[newDataFrame[col] > -2.5]
        makeHistogram(newDataFrame, col)

    #plot a line graph using the epa_state_year_frame.csv file
    #epa_state_year_frame.csv shows sum of chemical release per state per year
    #line graph shows puts the states into regional categories, averages their release sums for each year
    lineGraphByRegion(myDataFrame3, ['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'])

    #testAccuracy(myDataFrame1)

def runStats(myDataFrame, numCols):
    #create lists for each of the measurements
    means = []
    medians = []
    stdDevs = []
    #loop through and append measurements for each column to respective list
    for col in numCols:
        means.append(myDataFrame[col].mean())
        medians.append(myDataFrame[col].median())
        stdDevs.append(myDataFrame[col].std())
    #print the results
    print('columns', numCols)
    print('means', means)
    print('medians', medians)
    print('stddevs', stdDevs)
    #also print a table that has mean and stddev
    print(myDataFrame.describe())
    print()
    #return measurements in cased needed later
    return (means, medians, stdDevs)

def equiWidthBinning(myDataFrame, col, numBins):
    #puts data into bins of equal width
    minVal = myDataFrame[col].min() - 1
    maxVal = myDataFrame[col].max() + 1

    step = (maxVal - minVal) / numBins
    bins =  np.arange(minVal, maxVal + step, step)

    equiWidthBins = np.digitize(myDataFrame[col], bins)

    binCounts = np.bincount(equiWidthBins)
    print("\n\nBins for " + col + " are: \n ", equiWidthBins)
    print("\nBin count is ", binCounts)

    return myDataFrame[col]

def equiDepthBinning(myDataFrame, col, numBins):
    #puts data into bins of equal depth
    equiDepthBins = pd.qcut(myDataFrame[col], q=numBins)

    binCounts = np.bincount(equiDepthBins)
    print("\n\nBins for " + col + " are: \n ", equiDepthBins)
    print("\nBin count is ", binCounts)

    colName = col + ' (ED_BIN)'
    myDataFrame[colName] = equiDepthBins
    return myDataFrame[col]

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

def makeHistogram(myDataFrame, col):
    #plot the data as a histogram
    binnedData = equiWidthBinning(myDataFrame, col, 10)
    plt.figure(1)
    binnedData.hist()
    plt.title(col + ' Distribution')
    plt.xlabel(col)
    plt.savefig('epa_'+ col + '_histogram.png')
    plt.show()

def makeScatterplot(myDataFrame, xCol, yCol):
    #first find linear regression of data and linear regression coefficient
    linReg = LinearRegression()
    linReg.fit(myDataFrame[xCol].values.reshape(-1, 1), myDataFrame[yCol])
    linCoef = linReg.score(myDataFrame[xCol].values.reshape(-1, 1), myDataFrame[yCol])

    #plot data as a scatter plot
    plt.figure(1)
    plt.scatter(myDataFrame[xCol], myDataFrame[yCol], s=50)
    plt.title(xCol + ' vs. ' + yCol)
    plt.xlabel(xCol)
    plt.ylabel(yCol)

    #add linear regression lines
    plt.plot(myDataFrame[xCol], linReg.predict(myDataFrame[xCol].values.reshape(-1, 1)))
    plt.savefig('epa_'+ xCol + '-' + yCol + '_scatter.png')
    plt.show()

    #return the linear regression coefficient for the data
    return linCoef

def normalizeData(myDataFrame, colList):
    #use scikit learn's normalize method to convert categorical values to numerical values
    myDataFrame[colList] = normalize(myData[colList])
    return myDataFrame

def normalizeByFeature(myDataFrame, toSortBy, toNormalize):
	for i in myDataFrame[toSortBy].unique():
		#for z-score standardizing
		stdv_value = myDataFrame.loc[myDataFrame[toSortBy] == i][toNormalize].std()
		mean_value = myDataFrame.loc[myDataFrame[toSortBy] == i][toNormalize].mean()
		myDataFrame.loc[myDataFrame[toSortBy] == i, toNormalize] = (myDataFrame.loc[myDataFrame[toSortBy] == i, toNormalize] - mean_value)/stdv_value
	return myDataFrame

def addRegionCat(myDataFrame):
    #adds a column for region
    index = 0
    regions = []
    for i in myDataFrame['STATE_ABBR']:
        #categorizes each state abbr by region
        regions.append(categorize(i))
    myDataFrame['REGION'] = regions
    return myDataFrame

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

def anovaTest(myDataFrame, var):
    #make lists to hold data for anova
    east = []
    cent = []
    mont = []
    pacf = []
    #make list to hold avg value of var per state across all years
    stateLists = {}

    for i in myDataFrame['STATE_ABBR'].unique():
        #(count, sum)
        stateLists[i] = (0,0)

    index = 0
    for i in myDataFrame[var]:
        state = myDataFrame['STATE_ABBR'][index]
        #increase the count for that state, increase the sum of var for that state
        stateLists[state] = (stateLists[state][0]+1,stateLists[state][1]+i)
        index+=1

    for i in stateLists:
        #categorize the state keys in stateLists
        if categorize(i) is 'EAST':
            #append the avg (sum/count) to respective regional list
            east.append(stateLists[i][1]/stateLists[i][0])
        elif categorize(i) is 'CENT':
            cent.append(stateLists[i][1]/stateLists[i][0])
        elif categorize(i) is 'MONT':
            mont.append(stateLists[i][1]/stateLists[i][0])
        else:
            pacf.append(stateLists[i][1]/stateLists[i][0])

    #run an anova to get f and p
    fvalue, pvalue = stats.f_oneway(east, cent, mont, pacf)
    return (fvalue, pvalue)

def lineGraphByRegion(myDataFrame, years):
    #make lists to hold data for graphs
    east = []
    cent = []
    mont = []
    pacf = []
    #make sums and counts to find averages
    eastSum = centSum = montSum = pacfSum = 0
    eastCount = centCount = montCount = pacfCount = 0

    #for each year
    for year in years:
        index = 0
        for i in myDataFrame[year]:
            #find respective regional category for state abbrev
            #add state data to that region's sum, increase count
            if categorize(myDataFrame.iloc[:,0][index]) is 'EAST':
                eastSum += i
                eastCount += 1
            elif categorize(myDataFrame.iloc[:,0][index]) is 'CENT':
                centSum += i
                centCount += 1
            elif categorize(myDataFrame.iloc[:,0][index]) is 'MONT':
                montSum += i
                montCount += 1
            else:
                pacfSum += i
                pacfCount += 1
            index+=1
        #add yearly average of sum_rel_est to respective list
        east.append(eastSum/eastCount)
        cent.append(centSum/centCount)
        mont.append(montSum/montCount)
        pacf.append(pacfSum/pacfCount)

    plt.plot(years, east, label='EAST')
    plt.plot(years, cent, label='CENT')
    plt.plot(years, mont, label='MONT')
    plt.plot(years, pacf, label='PACF')
    plt.legend(loc='upper right')
    plt.title('Average SUM_REL_EST per Region per Year')
    plt.show()

def testAccuracy(myDataFrame):
    #find number of columns in the dataframe
    col = len(myDataFrame.columns)-1
    valueArray = myDataFrame.values
    #x values should include all columns
    X = valueArray[:, 0:col]
    #y value is last column
    Y = valueArray[:, col]
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
    print(classification_report(Y_validate, predictions))

main()
