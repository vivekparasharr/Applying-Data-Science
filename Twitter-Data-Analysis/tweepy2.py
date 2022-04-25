# -*- coding: utf-8 -*-


# Streaming
# before you can start writing a stream listener script, following libraries have to be imported
# time library will be used create a time-out feature for the script
# os library will be used to set your working directory
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import os

# defining variables

consumer_key = '----'
consumer_secret = '----'
access_token = '----'
access_secret = '----'


# these variables will be used in the stream listener by being feed into the tweepy objects
start_time = time.time() #grabs the system time
keyword_list = ['twitter'] #track list

# just a test to print start_time
print(start_time)


# http://stats.seandolinar.com/collecting-twitter-data-using-a-python-stream-listener/

# Listener Class Override

# METHOD 1
# modify the StreamListener class by creating a child class to output the data into a .csv file
# rewrite the actions taken when the StreamListener instance receives data [the tweet JSON]
# open an output file, writes the JSON data as text, inserts a blank, and closes the document
class listener(StreamListener): 
	def __init__(self, start_time, time_limit=60): 
		self.time = start_time
		self.limit = time_limit 
	def on_data(self, data): 
		while (time.time() - self.time) < self.limit: 
			try: 
				saveFile = open('c:\Users\vivek_000\Documents\raw_tweets.json', 'a')
				saveFile.write(data)
				saveFile.write('\n')
				saveFile.close() 
				return True 
			except BaseException, e:
				print 'failed ondata,', str(e)
				time.sleep(5)
				pass 
		exit() 
	def on_error(self, status): 
		print statuses
		



####

class listener(StreamListener):

    def __init__(self, start_time, time_limit=1):

        self.time = start_time
        self.limit = time_limit

    def on_data(self, data):

        while (time.time() - self.time) < self.limit:
            try:
                tweet = json.loads(data)
                user_name = tweet['user']['name']
                tweet_count = tweet['user']['statuses_count']
                text = tweet['text']
                saveFile = open('raw_tweets.json', 'a')
                saveFile.write(text.encode('utf8'))
                saveFile.write('\n')
                saveFile.close()

                return True
            except BaseException, e:
                print 'failed ondata,', str(e)
                time.sleep(5)
                pass
        exit()

    def on_error(self, status):
        print statuses
        
####


# method 2
# alternative to csv output file
# eliminate the need for a .csv file and insert the tweet directly into a MongoDB database

from pymongo import MongoClient
import json


class listener(StreamListener):
    def __init__(self, start_time, time_limit=1):
        self.time = start_time
        self.limit = time_limit
    def on_data(self, data):
        while (time.time() - self.time) < self.limit:
            try:
                client = MongoClient('localhost', 27017)
                db = client['twitter_db']
                collection = db['twitter_collection']
                tweet = json.loads(data)
                collection.insert(tweet)
                return True
            except BaseException, e:
                print 'failed ondata,', str(e)
                time.sleep(5)
                pass
        exit()
    def on_error(self, status):
        print statuses
        
        
        


# Create an OAuthHandler instance to handle OAuth credentials
# Create a listener instance with a start time and time limit parameters passed to it
# Create an StreamListener instance with the OAuthHandler instance and the listener instance
auth = OAuthHandler(ckey, consumer_secret) #OAuth object
auth.set_access_token(access_token_key, access_token_secret)
twitterStream = Stream(auth, listener(start_time, time_limit=1)) #initialize Stream object with a time out limit
twitterStream.filter(track=keyword_list, languages=['en'])  #call the filter method to run the Stream Object




import sys
sys.exit()


