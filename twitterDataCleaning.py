
import pprint as pprint
import pandas as pd
import glob, os


def main():
	print("cleaning twitter data")
	queryList = ["melanoma"]
	#songDF = pd.read_csv('data.csv' , sep=',', encoding='latin1')

	for file in glob.glob("*.csv"):

		if "cleaned" not in file:
			print(file)
			twitterDF = pd.read_csv(file , sep=',', encoding='latin1')
			cleanData(twitterDF, queryList)
			saveData(twitterDF, file)


def cleanData(myData, queryList):

	delList = []

	for index, row in myData.iterrows():

		if any(x in row["Tweets"] for x in queryList):
			delList.append(index)

	print(delList)
	delList.sort(reverse = True)

	for i in delList:
		myData.drop(myData.index[i], inplace = True)

	myData.dropna(inplace = True)


def saveData(myData, fileName):

    fileName = fileName.replace('.csv', '')
    fileName = fileName + "_cleaned.csv"
    myData.to_csv(fileName)


if __name__ == '__main__':

    main()
    print("done")
