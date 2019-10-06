
import pprint as pprint
import pandas as pd
import glob, os


def main():
	print("cleaning twitter data")
	queryList = ["pollution", "smog", "chemicals", "toxins", "contamination", "emissions", "sewage"]
	#songDF = pd.read_csv('data.csv' , sep=',', encoding='latin1')

	for file in glob.glob("*.csv"):

		if "cleaned" not in file:
			print(file)
			twitterDF = pd.read_csv(file , sep=',', encoding='latin1')
			#cleanTweetText_Remove(twitterDF, queryList)
			cleanNullValue_Scoring(twitterDF)
			cleanTweetText_Scoring(twitterDF, queryList)
			saveData(twitterDF, file)


def cleanTweetText_Remove(myData, queryList):

	delList = []

	for index, row in myData.iterrows():

		isThere = 0

		for x in queryList:

			if x in row["tweet_text"]:
				isThere = isThere + 1

		if isThere == 0:
			delList.append(index)


	print(delList)
	delList.sort(reverse = True)

	for i in delList:
		myData.drop(myData.index[i], inplace = True)

	myData.dropna(inplace = True)

def cleanTweetText_Scoring(myData, queryList):

	delList = []
	tweetTermScore = []

	columnsList = myData.columns

	for index, row in myData.iterrows():

		isThere = 0

		for x in queryList:

			if x in row["tweet_text"]:
				isThere = isThere + 1

		tweetTermScore.append(str(float(isThere/len(queryList))))

	delList.sort(reverse = True)

	myData["TweetTermScore"] = tweetTermScore

def cleanNullValue_Scoring(myData):

	NullValueScore = []

	columnsList = list(myData.columns.values)
	columnsList.reverse()
	columnsList.pop()
	columnsList.reverse()
	print(columnsList)

	isEmpty = 0
	isNull = 0

	for index, row in myData.iterrows():

		for name in columnsList:
			if not row[name].strip(): #empty string
				isEmpty = isEmpty + 1
			if row[name] == "None" or row[name] == "null":
				isNull = isNull +1

		totalBad = isEmpty + isNull

		NullValueScore.append(str(float(totalBad/len(columnsList))))

	myData["NullValueScore"] = NullValueScore


def saveData(myData, fileName):

	fileName = fileName.replace('.csv', '')
	fileName = fileName + "_cleaned.csv"
	if not os.path.exists('DoneCleaning'):
		os.makedirs('DoneCleaning')
	myData.to_csv(r'DoneCleaning/'+fileName, index=False)

if __name__ == '__main__':

    main()
    print("done")
