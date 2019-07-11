import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

query = input('Digite para ver as opiniões: ')

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search(query, count=200)

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

sid = SentimentIntensityAnalyzer()

listy = []

for index, row in data.iterrows():
  ss = sid.polarity_scores(row["Tweets"])
  listy.append(ss)
  
se = pd.Series(listy)
data['polarity'] = se.values

ls = list(se)
tot = {'negativo': 0, 'positivo': 0}

for x in range(len(ls)):
  tot['positivo'] += ls[x]['pos']
  tot['negativo'] += ls[x]['neg']

fig1, ax1 = plt.subplots()
ax1.set_title('{} em {}'.format(query, tweets[0].created_at))
ax1.pie(list(tot.values()), labels=list(tot.keys()), autopct='%1.1f%%', startangle=90)
ax1.axis('equal')

mng = plt.get_current_fig_manager()
mng.set_window_title(query)

plt.show()
