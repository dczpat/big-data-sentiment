import tweepy

# from datetime import date, timedelta

consumer_key = "QlMOJqo8ceANzEClxdkO6UsGd"
consumer_secret = "MNgJk5aBXrTjUAUo5sDuSjL9lccofGcKJGYRPVQ8uytAnm8FQz"
access_token = "1078687032187047936-JiRgRrEzx2usmUhb3n5u59gx3SC9vd"
access_secret = "j5sD88wK8oFJrmKPtQTUN4hWg0kI3gWxqBP2aSSBJsvqe"
tweetsPerQry = 10
maxTweets = 50
hashtag = "#singapore"

authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_secret)
api = tweepy.API(
    authentication,
    wait_on_rate_limit=True,
)
tweetCount = 0
while tweetCount < maxTweets:
    newTweets = api.search_full_archive(
        label="prod", query="singapore", fromDate="202210011000", maxResults=30
    )

    if not newTweets:
        print("Tweet Habis")
        break

    for tweet in newTweets:
        print(tweet.text)

    tweetCount += len(newTweets)
