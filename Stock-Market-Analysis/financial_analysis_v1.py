# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:17:48 2019

@author: Vivek
"""

'''
No.	Company	Revenue
(billion US dollars)

Headquarters
1	Glencore	209.2	  Switzerland
2	BHP Billiton	69.4	 Australia
3	Rio Tinto	45.1	 United Kingdom
4	China Shenhua Energy	40	 China
5	Vale	33.2	 Brazil
'''

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

start = dt.datetime(2018, 1, 1)
end = dt.datetime.now()
c=['BHP', 'RIO', 'VALE']
f=['BHP.csv', 'RIO.csv', 'VALE.csv']

for i in range(3):
    df = web.DataReader(c[i], 'iex', start, end)
    df.to_csv(f[i])

df = pd.read_csv(f[0], parse_dates=True, index_col=0)

'''plotting'''

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()




