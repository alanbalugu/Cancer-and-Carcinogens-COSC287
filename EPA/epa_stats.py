import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pprint import pprint

def main():
    #read data from file into pandas dataframe
    myDataFrame1 = pd.read_csv("epa_data_state_releases_cleaned.csv", sep=',', encoding='latin1')
    #create list of names for columns with numerical data
    numCols1 = ['SUM_REL_EST','AVG_REL_EST','MIN_REL_EST','MAX_REL_EST',
        'STD_REL_EST','VAR_REL_EST','CLEAN_SCORE']
    #run stats tests on the data
    stats1 = runStats(myDataFrame1, numCols1)
    print(numCols1)
    print(stats1[0])
    print(stats1[1])
    print(stats1[2])
    print(stats1[3])
    #change categorical to numerical
    myDataFrame1 = catToNum(myDataFrame1, 'STATE_ABBR')
    print(myDataFrame1[:10])
    #for col in numCols1:
    #    makeHistogram(myDataFrame1, col)
    linCoefs = {}
    for colA in numCols1:
        for colB in numCols1:
            if colA is not colB:
                makeScatterplot(myDataFrame1, colA, colB)
                linCoef = findLinReg(myDataFrame1, colA, colB)
                linCoefs[(colA,colB)] = linCoef
    print (linCoefs)
    #read data from other epa file into pandas dataframe
    myDataFrame2 = pd.read_csv("epa_data_state_chems_and_releases_cleaned.csv", sep=',', encoding='latin1')
    #create list of names for columns with numerical data
    numCols2 = ['SUM_REL_EST','AVG_REL_EST','MIN_REL_EST','MAX_REL_EST',
        'STD_REL_EST','VAR_REL_EST','CLEAN_SCORE']
    #run stats tests on the data
    stats2 = runStats(myDataFrame2, numCols2)
    print(numCols2)
    print(stats2[0])
    print(stats2[1])
    print(stats2[2])
    print(stats2[3])

def runStats(myDataFrame, numCols):
    means = []
    medians = []
    modes = []
    stdDevs = []
    for col in numCols:
        means.append(myDataFrame[col].mean())
        medians.append(myDataFrame[col].median())
        modes.append(myDataFrame[col].mode())
        stdDevs.append(myDataFrame[col].std())
    return (means, medians, modes, stdDevs)

def equiWidthBinning(myDataFrame, col, numBins):
    minVal = myDataFrame[col].min() - 1
    maxVal = myDataFrame[col].max() + 1

    step = (maxVal - minVal) / numBins
    bins =  np.arange(minVal, maxVal + step, step)

    equiWidthBins = np.digitize(myDataFrame[col], bins)

    binCounts = np.bincount(equiWidthBins)
    print("\n\nBins are: \n ", equiWidthBins)
    print("\nBin count is ", binCounts)

    colName = col + ' (EW_BIN)'
    myDataFrame[colName] = equiWidthBins
    print(myDataFrame[:10])

def equiDepthBinning(myDataFrame, col, numBins):
    equiDepthBins = pd.qcut(myDataFrame[col], q=numBins)

    binCounts = np.bincount(equiDepthBins)
    print("\n\nBins are: \n ", equiDepthBins)
    print("\nBin count is ", binCounts)

    colName = col + ' (ED_BIN)'
    myDataFrame[colName] = equiDepthBins
    print(myDataFrame[:10])

def catToNum(myDataFrame, col):
    numDict = {}

    #find unique values for column
    uniqueList = pd.unique(myDataFrame[col])
    count = 0
    for item in uniqueList:
        #if value is ?, change to null
        if item is not np.nan:
            numDict[item] = count
            count+=1

    print(numDict)
    #replace all of the categorical values with their respective int values
    myData = myDataFrame.replace(numDict)
    return myDataFrame

def makeHistogram(myDataFrame, col):
    plt.figure(1)
    myDataFrame[col].hist()
    plt.title(col + ' Distribution')
    plt.xlabel(col)
    plt.savefig('epa_'+ col + '_histogram.png')
    plt.show()

def makeScatterplot(myDataFrame, xCol, yCol):
	plt.figure(1)
	plt.scatter(myDataFrame[xCol], myDataFrame[yCol], s=50)
	plt.title(xCol + ' vs. ' + yCol)
	plt.xlabel(xCol)
	plt.ylabel(yCol)

	plt.savefig('epa_'+ xCol + '-' + yCol + '_scatter.png')
	plt.show()

def findLinReg(myDataFrame, xCol, yCol):
    linReg = LinearRegression()
    linReg.fit(myDataFrame[xCol].values.reshape(-1, 1), myDataFrame[yCol])
    linCoef = linReg.score(myDataFrame[xCol].values.reshape(-1, 1), myDataFrame[yCol])
    print(xCol, yCol, linCoef)
    return linCoef

main()
