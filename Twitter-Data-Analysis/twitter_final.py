#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 18:57:12 2017

@author: vivekparashar
"""

########### step 1 - download data from twitter ##################

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "----"
access_token_secret = "----"
consumer_key = "----"
consumer_secret = "-----"

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    def on_data(self, data):
        if 'text' in data:
            file = open('/Users/vivekparashar/Documents/python dsp jupyter notebook/tweets_ignacio.txt', 'a')
            file.write(data) 
            file.close()
        return True
    def on_error(self, status):
        #print (status)
        return True

if __name__ == '__main__':
    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    #This line filter Twitter Streams to capture data by the keywords
    stream.filter(track=['trump','climate change','paris agreement'])

########### step 2 - put the data in a dataframe ##################

import json
import pandas as pd
import matplotlib.pyplot as plt

#read the tweet data from the file into a list
tweets_data_path = '/Users/vivekparashar/Documents/python dsp jupyter notebook/tweets_ignacio.txt'
tweets_data = [] #a list
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
    
print (len(tweets_data))  #print the number of tweets

#structure the tweets data into a pandas DataFrame
#creating an empty DataFrame called tweets
tweets = pd.DataFrame()

#add 2 columns to the tweets DataFrame called text and lang. text column contains the tweet
tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] = list(map(lambda tweet: tweet['lang'], tweets_data))

#add col about location which us nexter under user in tweet data
user = list(map(lambda tweet: tweet['user'], tweets_data))
tweets['loc']=list(map(lambda tweet_user: tweet_user['location'], user))

############## step 3 - charting data ########################

#create 2 charts: 
#The first one describing the Top 5 languages in which the tweets were written
tweets_by_lang = tweets['lang'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')

#The first one describing the Top 5 languages in which the tweets were written
tweets_by_loc = tweets['loc'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Location', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 locations', fontsize=15, fontweight='bold')
tweets_by_loc[:5].plot(ax=ax, kind='bar', color='red')

############### step 4 - tweets about the topics ################

import re 

#following function returns True if a word is found in text, otherwise it returns False.
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

#add 3 columns to our tweets DataFrame
tweets['python'] = tweets['text'].apply(lambda tweet: word_in_text('python', tweet))
tweets['java'] = tweets['text'].apply(lambda tweet: word_in_text('java', tweet))
tweets['ruby'] = tweets['text'].apply(lambda tweet: word_in_text('ruby', tweet))

#number of tweets for python programming language
print (tweets['python'].value_counts()[True])

#getting programming languages and tweets by prog languages
prg_langs = ['python', 'java', 'ruby']
tweets_by_prg_lang = [tweets['python'].value_counts()[True], tweets['java'].value_counts()[True], tweets['ruby'].value_counts()[True]]

#plotting above data
x_pos = list(range(len(prg_langs)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width, alpha=1, color='g')
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: python vs. java vs. ruby (Raw data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.grid()

############ step 5 - targeting relevant tweets ####################

tweets['programming'] = tweets['text'].apply(lambda tweet: word_in_text('programming', tweet))
tweets['tutorial'] = tweets['text'].apply(lambda tweet: word_in_text('tutorial', tweet))
tweets['relevant'] = tweets['text'].apply(lambda tweet: word_in_text('programming', tweet) or word_in_text('tutorial', tweet))

print (tweets['relevant'].value_counts()[True])
print (tweets[tweets['relevant'] == True]['python'].value_counts()[True])
print (tweets[tweets['relevant'] == True]['java'].value_counts()[True])
print (tweets[tweets['relevant'] == True]['ruby'].value_counts()[True])

tweets_by_prg_lang = [tweets[tweets['relevant'] == True]['python'].value_counts()[True], 
                      tweets[tweets['relevant'] == True]['java'].value_counts()[True], 
                      tweets[tweets['relevant'] == True]['ruby'].value_counts()[True]]

#plotting above data
x_pos = list(range(len(prg_langs)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width,alpha=1,color='g')
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: python vs. java vs. ruby (Relevant data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.grid()

############### step 6 - sentiment analysis #######################

# pip install textblobs
# python -m textblob.download_corpora

from textblob import TextBlob

tweets['text'][0] # first tweet

##a way to loop through pandas dataframe row by row
for index, row in tweets.iterrows():
    print (row["lang"],' - ',row["lang"])      

##Sentiment Analysis¶ - adding polarity and subjectivity columns
# The polarity score is a float within the range [-1.0, 1.0]. 
# The subjectivity is a float within the range [0.0, 1.0] 
# where 0.0 is very objective and 1.0 is very subjective.

tweets['pol'] = ""      
tweets['subj'] = "" 
for index, row in tweets.iterrows():   
    #print (tweets['lang'][index])
    tb=TextBlob(tweets['text'][index])
    tweets['pol'][index]=tb.sentiment.polarity
    tweets['subj'][index]=tb.sentiment.subjectivity
     
print(tweets)    

##### Additional things that can be done with textblobs    
TextBlob(tweets['text'][0]).tags #Part-of-speech Tagging
TextBlob(tweets['text'][0]).noun_phrases #Noun Phrase Extraction
TextBlob(tweets['text'][0]).words.count('ruby') # this could be used instead of re
TextBlob(tweets['text'][0]).noun_phrases.count('python')
##Tokenization - break TextBlobs into words or sentences
TextBlob(tweets['text'][0]).words
TextBlob(tweets['text'][0]).sentences
for sentence in TextBlob(tweets['text'][0]).sentences:
    print(sentence.sentiment)
        
################ step 7- attempt a word cloud ########################

#pip install git+git://github.com/amueller/word_cloud.git

from wordcloud import WordCloud

# text from all relevant + python tweets
text=""
for index, row in tweets.iterrows(): 
    if tweets['python'][index]==[True] and tweets['relevant'][index] == True:
        text=text+' '+tweets['text'][index]

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


### remember that for ruby we have a lot of irrelevant tweets
## lets see what those are about

# text from all irrelevant + ruby tweets
text=""
for index, row in tweets.iterrows(): 
    if tweets['ruby'][index]==[True] and tweets['relevant'][index] == False:
        text=text+' '+tweets['text'][index]

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



############ downloading stocks data ###########################


gticker='LON:EZJ'
import pandas_datareader as web
dfg = web.DataReader(gticker, 'google', '2017/5/29', '2017/6/9')

dfg #returned pandas dataframe
sa=dfg.reset_index().values #convert pandas dataframe to numpy array

#accessing dataframe                  
dfg.loc[:,['Volume']]
dfg.iloc[:,[4]]

dfg.index.dtype #dtype of index

dfg.columns.tolist() #enlists column names

import matplotlib.pyplot as plt
dfg.boxplot(column="Open",by="High") #box plot of data from pandas dataframe

plt.plot(sa[:,0],sa[:,2:3],label='High')
plt.plot(sa[:,0],sa[:,3:4],label='Low')
plt.legend()

plt.bar(sa[:,0],sa[:,2:3],label='High')

bins=[140,145,150,155,160,165,170,175,180,185,190] 
plt.hist(sa[:,2:3],bins,histtype='bar',rwidth=0.8) #define bins yourself
plt.hist(sa[:,2:3],bins=20,histtype='bar',rwidth=0.8) #just tell the number of bins

plt.scatter(sa[:,1:2],sa[:,2:3]) #scatter plot of data, using data from structured array


           
