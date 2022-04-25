
#install.packages("streamR")
#install.packages("ROAuth")
library(ROAuth)
library(streamR)

#create your OAuth credential
credential <- OAuthFactory$new(consumerKey='----',
                               consumerSecret='----',
                               requestURL='https://api.twitter.com/oauth/request_token',
                               accessURL='https://api.twitter.com/oauth/access_token',
                               authURL='https://api.twitter.com/oauth/authorize')

#authentication process
options(RCurlOptions = list(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl")))
download.file(url="http://curl.haxx.se/ca/cacert.pem", destfile="cacert.pem")
credential$handshake(cainfo="cacert.pem")


#function to actually scrape Twitter
filterStream( file.name="tweets_test.json",
              track="twitter", tweets=1000, oauth=cred, timeout=1, lang='en' )



