

#consumer key, consumer secret, access token, access secret.
consumer_key = '----'
consumer_secret = '----'
access_token = '----'
access_secret = '----'



import xlwt
import time
import json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob #https://textblob.readthedocs.org/en/dev/quickstart.html#sentiment-analysis
#from elasticsearch import Elasticsearch

# import twitter keys and tokens
#from config import *

# create instance of elasticsearch
#es = Elasticsearch()

# you can use RFC-1738 to specify the url
#es = Elasticsearch(['https://consumer_key:consumer_secret@localhost:443'])


class TweetStreamListener(StreamListener):
    def on_data(self, data):    # on success
        try:    
            tweet2=data.split(',"text":"')[1].split('","source"')[0] #we want left side so 0th element
            loc=data.split(',"location":"')[1].split('","url"')[0] 
            usr_desc=data.split(',"description":"')[1].split('","protected"')[0] 
            followers=data.split(',"followers_count":')[1].split(',"friends_count"')[0]
            friends=data.split(',"friends_count":')[1].split(',"listed_count"')[0] 
            favo_cnt=data.split(',"favourites_count":')[1].split(',"statuses_count"')[0] 
            stats_cnt=data.split(',"statuses_count":')[1].split(',"created_at"')[0] 
            created=data.split(',"created_at":')[1].split('","utc_offset"')[0] 
            timezn=data.split(',"time_zone":"')[1].split('","geo_enabled"')[0] 
    
            dict_data = json.loads(data)        # decode json
            tweet = TextBlob(dict_data["text"])        # pass tweet into TextBlob
            print tweet
            pol= tweet.sentiment.polarity        # output sentiment polarity           # The polarity score is a float within the range [-1.0, 1.0]
            if tweet.sentiment.polarity < 0:        # determine if sentiment is positive, negative, or neutral
                sentiment = "negative"
            elif tweet.sentiment.polarity == 0:
                sentiment = "neutral"
            else:
                sentiment = "positive"
            #print sentiment        # output sentiment
            subj= tweet.sentiment.subjectivity        # The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
            # add text and sentiment info to elasticsearch
            #saveThis=str(time.time())+'::'+tweet2+'::'+loc+'::'+usr_desc+'::'+followers+'::'+friends+'::'+favo_cnt+'::'+stats_cnt+'::'+created+'::'+timezn+'::'+pol+'::'+subj+'::'+sentiment #double colon is the separator because people might use single colon in their tweet
            #saveThis=str(time.time())+'::'+tweet2+'::'+loc+'::'+followers+'
            saveFile=open('twitDB99.csv','a') # a means append
            saveFile.write(tweet2)#(saveThis)
            saveFile.write('\n') #new line
            saveFile.write(sentiment)
            saveFile.write('\n') #new line
            saveFile.write('\n') #new line
            saveFile.write('\n') #new line

            saveFile.close()
            return(True)
        except BaseException, e:
                print 'failed ondata,',str(e)
                time.sleep(5)
    # on failure
    def on_error(self, status):
        print status

#if __name__ == '__main__':

    # create instance of the tweepy tweet stream listener
    #listener = TweetStreamListener()

    # set twitter keys/tokens
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # create instance of the tweepy stream
    stream = Stream(auth, TweetStreamListener())

    # search twitter for "congress" keyword
    stream.filter(track=['car'], languages=['en'])





