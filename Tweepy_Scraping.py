import tweepy
import pandas as pd


#Chen Zhi
api_key = "QlMOJqo8ceANzEClxdkO6UsGd"
api_secret_key = "MNgJk5aBXrTjUAUo5sDuSjL9lccofGcKJGYRPVQ8uytAnm8FQz"
access_token = "1078687032187047936-JiRgRrEzx2usmUhb3n5u59gx3SC9vd"
access_token_secret = "j5sD88wK8oFJrmKPtQTUN4hWg0kI3gWxqBP2aSSBJsvqe"

'''Nic (academic access pending)
api_key = "gQXXacRwuBZnikH87J6eIvrcU"
api_secret_key = "cMu0NzBqKXQWv2TFRRMGG69vNIly0HuEV2f5DqslKBo9tCvWbA"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAMfCiAEAAAAAv0e6aHvGJp4aPsKHm1SipI2VEG0%3DXaLNJBqAuVRbpnSp7rWaIjsiKeI2aVV6aeSHNyKXykUqp0F8KQ"
access_token = "912776727620878336-Utux0D1GAIXGpS60GvqhzPnKgqoX7CS"
access_token_secret = "DAMOpVqqY9bAp3DegN88dDdLgXFIP0tzud47p8N5hvoRO"
'''

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

try:
    api.verify_credentials()
    print("It works!")
except:
    print("It didn't work")

#Get Tweets using Hashtags
keywords = '#OpIran'
limit = 100000

keyword_tweets = tweepy.Cursor(
    api.search_tweets
    ,q=keywords
    ,lang='en'
    ,count=100
    ,tweet_mode='extended').items(limit)

columns = ['Time', 'User', 'Tweet']
data = []
for tweet in keyword_tweets:
  data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)
print(df)
df.to_csv('keyword_tweets_OpIran.csv')