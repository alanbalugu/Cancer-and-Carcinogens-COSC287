#Alan Balu, agb76

import pprint as pprint
import pandas as pd
import glob, os


def main():
	print("cleaning twitter data")
	queryList = ["pollution", "pollutant", "polluting", "greenhouse gas", "biohazard", "co2", "carbon dioxide", "dirty air", "dirty water", "ozone", "smog", "chemical", "toxin", "particulates", "contamination", "contaminant", "emission", "sewage", "waste", "pesticide", "carcinogen"]
	#songDF = pd.read_csv('data.csv' , sep=',', encoding='latin1')

	#for each .csv file in the directly, clean the twitter data
	for file in glob.glob("*.csv"):

		#only clean the files that are not already cleaned
		if "cleaned" not in file:
			print(file)
			twitterDF = pd.read_csv(file, sep=',', encoding="utf-8")

			#convert the text of tweets to lowercase
			twitterDF["tweet_text"] = twitterDF['tweet_text'].str.lower()
			#cleanTweetText_Remove(twitterDF, queryList)

			#add the scoring for null/bad values in data
			cleanNullValue_Scoring(twitterDF)

			#add scoring for tweet text containing pollution related terms
			cleanTweetText_Scoring(twitterDF, queryList)

			#save the data
			saveData(twitterDF, file)


#function to clean the tweet text and remove rows that don't have desired terms
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


#function to add cleanliness scoring
def cleanTweetText_Scoring(myData, queryList):

	tweetTermScore = []

	columnsList = myData.columns

	#iterate through dataframe and check if each query term is in the tweet_text attributes
	for index, row in myData.iterrows():

		isThere = 0

		for x in queryList:

			if x in row["tweet_text"]:
				isThere = isThere + 1

		#add the score based on the number of query terms and the number of terms found in string
		tweetTermScore.append(str(float(isThere/len(queryList))))

	myData["TweetTermScore"] = tweetTermScore

#function to add bad/null value scoring
def cleanNullValue_Scoring(myData):

	NullValueScore = []

	#remove the first item in the list of columns (row number column)
	columnsList = list(myData.columns.values)
	columnsList.reverse()
	columnsList.pop()
	columnsList.reverse()
	print(columnsList)

	#iterate through dataframe and check if there are null/bad values
	for index, row in myData.iterrows():

		isEmpty = 0
		isNull = 0
		hasRT = 0

		#check for empty string, check if null or None is in the data 
		for name in columnsList:
			if not row[name].strip(): #empty string
				isEmpty = isEmpty + 1
			if row[name] == "None" or row[name] == "null":
				isNull = isNull +1

		#check if the tweet is a retweet
		if "RT @" in row["tweet_text"]:
			hasRT = hasRT + 1

		#score total (retweets are not desirable at the moment)
		totalBad = isEmpty + isNull + hasRT

		#add the bad value score to the list
		NullValueScore.append(str(float(totalBad/len(columnsList))))

	#add the bad value score column
	myData["BadValueScore"] = NullValueScore

#save the data with _cleaned added to the file name
def saveData(myData, fileName):

	fileName = fileName.replace('.csv', '')
	fileName = fileName + "_cleaned.csv"
	if not os.path.exists('DoneCleaning'):
		os.makedirs('DoneCleaning')
	myData.to_csv(r'DoneCleaning/'+fileName, index=False)

if __name__ == '__main__':

    main()
    print("done")
