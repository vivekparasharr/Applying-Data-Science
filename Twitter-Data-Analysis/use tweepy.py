# -*- coding: utf-8 -*-

# http://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
# connecting to twitter
# these variables will be used by being feed into the tweepy objects
import tweepy
from tweepy import OAuthHandler
 
consumer_key = '----'
consumer_secret = '----'
access_token = '----'
access_secret = '----'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


# get your own timeline
from tweepy import Cursor
for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text) 
    

# get timeline of any user
# user defined function to print output from cursor
def process_status(sta):
    print sta.text

# function to get timeline data for a specific user
for status in tweepy.Cursor(api.user_timeline, id="DRUNKHULK").items(1):
    # process status here
    #process_status(status)
    print(status.text) 


# get json of a tweet from your timeline or anyone elses timeline
import json
for tweet in tweepy.Cursor(api.user_timeline, id="DRUNKHULK").items(1):
    print(tweet.json)


# a list of all your followers
for friend in tweepy.Cursor(api.friends).items():
    print(friend.name)

for follower in api.followers_ids('vivek_sscbs'):
    print api.get_user(follower).screen_name



# Streaming
# before you can start writing a stream listener script, following libraries have to be imported
# time library will be used create a time-out feature for the script
# os library will be used to set your working directory
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import os

# keep the connection open, and gather all the upcoming tweets about a particular event
# you can use the command wc -l python.json from a Unix shell to know how many tweets you’ve gathered
from tweepy import Stream
from tweepy.streaming import StreamListener
 
class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#python'])


# these variables will be used in the stream listener by being feed into the tweepy objects
start_time = time.time() #grabs the system time
keyword_list = ['twitter'] #track list

# just a test to print start_time
print(start_time)


# http://stats.seandolinar.com/collecting-twitter-data-using-a-python-stream-listener/

# modify the StreamListener class by creating a child class to output the data into a .csv file
# rewrite the actions taken when the StreamListener instance receives data [the tweet JSON]
#Listener Class Override
class listener(StreamListener): 
	def __init__(self, start_time, time_limit=60): 
		self.time = start_time
		self.limit = time_limit 
	def on_data(self, data): 
		while (time.time() - self.time) < self.limit: 
			try: 
				saveFile = open('raw_tweets.json', 'a')
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
		

# open an output file, writes the JSON data as text, inserts a blank, and closes the document
saveFile = open('raw_tweets.json', 'a')
saveFile.write(data)
saveFile.write('\n')
saveFile.close()


# alternative to csv output file
# eliminate the need for a .csv file and insert the tweet directly into a MongoDB database



# Create an OAuthHandler instance to handle OAuth credentials
# Create a listener instance with a start time and time limit parameters passed to it
# Create an StreamListener instance with the OAuthHandler instance and the listener instance
auth = OAuthHandler(ckey, consumer_secret) #OAuth object
auth.set_access_token(access_token_key, access_token_secret)
twitterStream = Stream(auth, listener(start_time, time_limit=20)) #initialize Stream object with a time out limit
twitterStream.filter(track=keyword_list, languages=['en'])  #call the filter method to run the Stream Object











