from pprint import pprint
from tweepy.streaming import StreamListener
import tweepy as tw
import pandas as pd
from datetime import datetime

#Variables that contains the user credentials to access Twitter API 

consumer_key="8ba7KKye3FQ65PIXvyg7sf7Hg"
consumer_secret="xs7cLcQqu54MzTVFlZpqYHg2roabjqEmGDyuc750y3t80mFLiC"
access_key="989511788919230464-Wog5fx5tQf6pThOlArGcMWuyvHSYe9c"
access_secret="CBoO595aL8pu3YlP7ZtzRQ0J7mBo7YSY0jpmQ7p4a9qRT"


def main():
    print("gathering recent twitter data")

    geoCodeList = ["38.920704,-77.030325,30km"]
    search_query = "melanoma OR cancer"
    date_since = "2019-8-6"
    geoCode = "38.920704,-77.030325,30km"
    numItems = 200

    for geo in geoCodeList:

        searchTerm = search_query + " -filter:retweets"

        outputFrame = pd.DataFrame()

        tweets = getTweets(searchTerm, date_since, geo, numItems)

        formatData(outputFrame,tweets)

        saveData(outputFrame)


def saveData(outputFrame):

    myFileName="TwitterData_" + str(datetime.now().strftime('%Y.%m.%d_%H.%M.%S')) + ".csv"
    outputFrame.to_csv(myFileName)

def formatData(outputFrame, tweets):

    tweetLocations = []
    tweetText = []
    tweetDates = []

    for tweet in tweets:
        print(tweet.user.location)
        tweetLocations.append(str(tweet.user.location))
        #print(tweet.text)
        tweetText.append(str(tweet.text))
        #print(tweet.created_at)
        tweetDates.append(str(tweet.created_at))

    outputFrame["Tweets"] = tweetText
    outputFrame["Locations"] = tweetLocations
    outputFrame["Dates"] = tweetDates

    pprint(outputFrame)


def getTweets(search_query, date_since, geoCode, numItems):

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    tweets = tw.Cursor(api.search,
                  q=search_query,
                  lang="en",
                  since=date_since,
                  geocode = geoCode
                  ).items(numItems)

    # for tweet in tweets:
    #     print(tweet.user.location)
    #     print(tweet.text)
    #     print(tweet.created_at)

    return tweets


if __name__ == '__main__':

    main()
    print("done")

    #This handles Twitter authetification and the connection to Twitter Streaming API
    '''l = StdOutListener()
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    stream = tw.Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    #stream.filter(track=['Donald Trump'])
    stream.filter(locations=[-75.8634,38.4623,-75.2729,39.8544])'''