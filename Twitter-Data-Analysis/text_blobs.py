#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 11:06:25 2017

@author: vivekparashar
"""

# pip install textblobs
# python -m textblob.download_corpora

from textblob import TextBlob

tweets['text'][0] # first tweet


      
teets_b=TextBlob(tweets['text'][0])
teets_b.sentiment
teets_b.sentiment.polarity
teets_b.noun_phrases

for sentence in teets_b.sentences:
    print(sentence.sentiment)


for i in range(1000):
    print (TextBlob(tweets['text'][i]).sentiment)
        