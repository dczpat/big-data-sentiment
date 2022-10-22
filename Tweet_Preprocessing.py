import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

file = 'C:/Users/NTW/Downloads/CS5344 Big-Data Analytics Technology/Project/keyword_tweets_UkraineRussianWar_70k_rows.csv'

data = pd.read_csv(file)

data = data.drop(columns=data.columns[[0,2]], axis=1)

print(data.shape)

word_tokens = set(stopwords.words('english'))

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in word_tokens]
    temp = " ".join(word for word in temp)
    return temp

#tweet_column = pd.DataFrame(data['Tweet'])
#tweet_column = clean_tweet(tweet_column)
data['Tweet'] = data['Tweet'].apply(lambda x: clean_tweet(x))

print(data.head())

data.to_csv('C:/Users/NTW/Downloads/CS5344 Big-Data Analytics Technology/Project/keyword_tweets_UkraineRussianWar_70k_rows_clean.csv')
