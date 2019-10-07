import tweepy
from pprint import pprint
import pandas as pd
from datetime import datetime
import time


#API keys and access token for OAuth process
consumer_key="8ba7KKye3FQ65PIXvyg7sf7Hg"
consumer_secret="xs7cLcQqu54MzTVFlZpqYHg2roabjqEmGDyuc750y3t80mFLiC"
access_token="989511788919230464-Wog5fx5tQf6pThOlArGcMWuyvHSYe9c"
access_token_secret="CBoO595aL8pu3YlP7ZtzRQ0J7mBo7YSY0jpmQ7p4a9qRT"

#saves data to .csv file with name and timestanp
def saveData(outputFrame, handle):

    myFileName="TwitterData_" + str(datetime.now().strftime('%Y.%m.%d_%H.%M.%S')) + handle + ".csv"
    outputFrame.to_csv(myFileName, mode= 'a', encoding='utf-8-sig')

#formats the data into columns into a pandas dataframe
def formatData(outputFrame, tweets):

	tweetText = []
	tweetLocation = []
	tweetDate = []
	tweetHandle = []
	
	for tweet in tweets:
		print(tweet.full_text + " >> " + tweet.user.screen_name)
		tweetText.append(str(tweet.full_text))
		tweetLocation.append(str(tweet.place))
		tweetDate.append(str(tweet.created_at))
		tweetHandle.append(str(tweet.user.screen_name))

	print("got this many tweets: ", len(tweetText))

	outputFrame["tweet_date"] = tweetDate
	outputFrame["tweet_location"] = tweetLocation
	outputFrame["tweet_text"] = tweetText
	outputFrame["tweet_handle"] = tweetHandle

#calls the tweepy wrapper for the twitter API and gets tweets, returns that JSON data
def getTweets(handle):
	# OAuth process, using the keys and tokens
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	# Creation of the actual interface, using authentication
	api = tweepy.API(auth)

	tweets = tweepy.Cursor(api.user_timeline, screen_name=handle, tweet_mode="extended", include_retweets = False).items()

	return tweets

#driver to get twitter data for each handle
def main():
	
	handleFile = open("twitterhandles.txt", 'r')

	handleList0 = handleFile.readlines()
	handleList = []

	for handle in handleList0:
		handleList.append(str(handle.strip()))

	print(handleList)

	for handle in handleList:
		       
		outputFrame = pd.DataFrame()

		tweets = getTweets(handle)
		formatData(outputFrame, tweets)
		saveData(outputFrame, handle)
		time.sleep(60)



if __name__ == '__main__':

    main()
    print("done")