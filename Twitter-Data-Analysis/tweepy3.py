

# http://pythonprogramming.net/twitter-api-streaming-tweets-python-tutorial/
# http://www.tulane.edu/~howard/CompCultES/twitter.html
# http://stats.seandolinar.com/collecting-twitter-data-using-a-python-stream-listener/

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time


#consumer key, consumer secret, access token, access secret.
consumer_key = '----'
consumer_secret = '----'
access_token = '----'
access_secret = '----'


#printing stream of data
class listener(StreamListener):
    def on_data(self, data):
        print(data)
        return(True)
    def on_error(self, status):
        print status

		
#saving stream of data in csv
class listener(StreamListener):
    def on_data(self, data):
        try:
            print(data)
            saveFile=open('twitDB_91.csv','a') # a means append
            saveFile.write(data)
            saveFile.write('\n') #new line
            saveFile.close()
            return(True)
        except BaseException, e:
                print 'failed ondata,',str(e)
                time.sleep(5)
    def on_error(self, status):
        print status


#saving stream of data in csv - only specific attributes
class listener(StreamListener):
    def on_data(self, data):
        try:
            tweet=data.split(',"text":"')[1].split('","source')[0] #we want left side so 0th element
            print tweet
            loc=data.split(',"location":"')[1].split(',"url')[0] #we want left side so 0th element
            print loc
            saveThis=str(time.time())+'::'+tweet+'::'+loc #double colon is the separator because people might use single colon in their tweet
            saveFile=open('twitDB93.csv','a') # a means append
            saveFile.write(saveThis)
            saveFile.write('\n') #new line
            saveFile.close()
            return(True)
        except BaseException, e:
                print 'failed ondata,',str(e)
                time.sleep(5)
    def on_error(self, status):
        print status


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)


twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])




