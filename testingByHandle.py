import tweepy

# Consumer keys and access tokens, used for OAuth
consumer_key="8ba7KKye3FQ65PIXvyg7sf7Hg"
consumer_secret="xs7cLcQqu54MzTVFlZpqYHg2roabjqEmGDyuc750y3t80mFLiC"
access_token="989511788919230464-Wog5fx5tQf6pThOlArGcMWuyvHSYe9c"
access_token_secret="CBoO595aL8pu3YlP7ZtzRQ0J7mBo7YSY0jpmQ7p4a9qRT"

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)

x = 0

for status in tweepy.Cursor(api.user_timeline, screen_name='@realDonaldTrump', tweet_mode="extended").items():
    print(status.full_text)
    x = x+1


print("total tweets: ", x)