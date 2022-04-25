# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:04:29 2019

@author: Vivek
"""


import datetime as dt
import pandas as pd
import pandas_datareader.data as web

'''getting the data'''
print('Getting the list of companies')
n = 20   # number of companies you want to pull the data for
df = pd.read_html('http://www.miningfeeds.com/home-mining-report-all-countries')[0] #pulling the company names
df = df.sort_values(by=['Volume'], ascending=False)  #sorting by volume, to get top n companies with highest volume

df = df[0:n][['Company','Ticker']].reset_index(drop=True) #get the tickers of the companies
c = list(df['Ticker']) #because we need to pass a list of datareader
for i in range(0,len(c)):
    c[i] = c[i].split('.')[0] #select the name before the .

print('Downloading actual Volume data - first 6 months of 2018')
#downloading actual volume data - first 6 months of 2018
start = dt.datetime(2018, 1, 1)
end = dt.datetime(2018, 6, 30) #datetime.now()

df2 = []
for i in range(0,len(c)):
    try:
        df2.append(web.DataReader(c[i], 'iex', start, end))
    except:
        c[i] = ''

d=[]
for i in range(0,len(c)):     
    if c[i]!='':
        d.append(c[i])

'''adding name column'''
for i in range(0,len(df2)):
    #df2[i]['7ma'] = df2[i]['open'].rolling(window=7).mean()
    df2[i]['name'] = d[i]

print('Preparing data for plotting')
''' creating the final data frame '''
for i in range(0,len(df2)):
    if i == 0:
        df3 = pd.DataFrame(df2[i]['open']).rename(columns={'open':d[i]})
    else:
        df3 = df3.join(pd.DataFrame(df2[i]['open']).rename(columns={'open':d[i]}))
    
'''plotting'''
print('Plotting')
def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
import matplotlib as matplotlib


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
rgb = [[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],[0,255,128],[0,255,255],[0,128,255],[0,0,255]]

for i in range(0,len(df3.columns)):
    
    if i+1<=9:
        plt.subplot(3,3,i+1)
        plt.plot(pd.to_datetime(df3.index), df3.iloc[:,i],color=rgb_to_hex(rgb[i][0],rgb[i][1],rgb[i][2] ))
        plt.legend()     
    
plt.show()
