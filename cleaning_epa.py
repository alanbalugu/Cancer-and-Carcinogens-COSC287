import numpy as np
import pandas as pd
from pprint import pprint

def main():
    #read data from file into pandas dataframe
    myData1 = readFile("epa_data_state_releases.csv")
    myDataFrame1 = pd.read_csv("epa_data_state_releases.csv", sep=',', encoding='latin1')
    #score each row based on how clean the data is: measurement = how many nonzero columns
    myDataFrame1 = rateCleanness(myDataFrame1)
    #drop rows with clean scores that do not reach a certain threshold value
    myDataFrame1 = dropData(myDataFrame1, 0.75)
    #print cleaned data to csv
    printData(myDataFrame1, "epa_data_state_releases_cleaned.csv")

    myData2 = readFile("epa_data_state_chems_and_releases.csv")
    myDataFrame2 = pd.read_csv("epa_data_state_chems_and_releases.csv", sep=',', encoding='latin1')
    myDataFrame2 = rateCleanness(myDataFrame2)
    myDataFrame2 = dropData(myDataFrame2, 0.75)
    printData(myDataFrame2, "epa_data_state_chems_and_releases_cleaned.csv")

def readFile(filename):
    temp = []
    inFile = open(filename, "r")
    line = inFile.readline()
    while (line):
    	if line.endswith('\n'):
                   #remove the new line symbol from the line
        	    line = line[:-1]
    	temp.append(line)
    	line = inFile.readline()
    inFile.close()
    return temp

def rateCleanness(myDataFrame):
    scoresColumn = []
    totalColumns = 0
    #loop through column headers in data frame to determine number of columns
    for i in myDataFrame:
        totalColumns+=1
    #loop through rows in data frame, sum number of nonzero columns
    for index,row in myDataFrame.iterrows():
        score = 0
        for i in row:
            if (i != 0):
                score+=1
        #collect cleanness score: number of nonzero columns / total number of columns
        scoresColumn.append(str(score/totalColumns))
    #add column to data frame to display cleanness scores
    myDataFrame['CLEAN_SCORE'] = scoresColumn
    return myDataFrame

def dropData(myDataFrame, threshold):
    index = []
    count = 0
    #loop through all values in the clean score column
    for i in myDataFrame['CLEAN_SCORE']:
        #if the clean score is less than the threshold value, append to index list
        if float(i) < threshold:
            index.append(count)
        count+=1
    #drop all values in index list
    return myDataFrame.drop(index)

def printData(myDataFrame, filename):
    #write to csv file
    myDataFrame.to_csv(filename)

main()
