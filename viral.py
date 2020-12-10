import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot

#import data
all_tweets = pd.read_json("random_tweets.json", lines=True)

#Discover data set
#The total number of tweets in the dataset.
#print(len(all_tweets))
#The columns, or features, of the dataset.
#print(all_tweets.columns)
#The text of the first tweet in the dataset.
#print(all_tweets.loc[0]['text'])
#The user that the text of the first tweet in the dataset.
#print(all_tweets.loc[0]['user'])
#The location which the user taht the text of the first tweet in the dataset.
#print(all_tweets.loc[0]['user']['location'])
#Print the user here and the user's location here.


# Defining median of our data
median_retweets = all_tweets["retweet_count"].median()

# Defining Viral Tweets
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > median_retweets, 1, 0)

# Making Features
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
all_tweets['hashtag_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis=1)
#all_tweets['links_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('http'),axis=1)
#all_tweets['words_count'] = all_tweets.apply(lambda tweet: tweet['text'].split(),axis=1)


labels = all_tweets['is_viral']
data = all_tweets[['tweet_length','followers_count','friends_count','hashtag_count']]
#print(all_tweets["hashtag_count"])

#Normalizing Data
scaled_data = scale(data,axis=0)

#Creating the Training Set and Test Set

train_data, test_data, train_labels, test_labels=train_test_split(scaled_data,labels, test_size=0.2, random_state =1)

# Using the Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_data, train_labels )

# Score
print(classifier.score(test_data, test_labels))

#Choosing K

scores = []
for k in range (1,200):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels )
    scores.append(classifier.score(test_data, test_labels))

#plt.plot(range(1,200), scores)
#plt.xlabel("K Value")
#plt.ylabel("Our Scores")
#plt.title("Choosing K Parameter")
#plt.show()
# Best k parametere is about 37

