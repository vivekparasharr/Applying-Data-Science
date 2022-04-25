
'''
Moving Averages: These strategies involve entering into long positions when a short-term 
moving average crosses above a long-term moving average, and entering short positions 
when a short-term moving average crosses below a long-term moving average.
'''

'''
n-day simple moving average = sum(close price for n-days) / n
Short-term averages respond quickly to changes in the price of the underlying security, while long-term averages are slower to react.

reading moving averages:
1. if the price is above a moving average, the trend is up. If the price is below a moving average, the trend is down
2. If the simple moving average points up, this means that the security's price is increasing. If it is pointing down, it means that the security's price is decreasing. 
3. If a shorter-term simple moving average is above a longer-term average, an uptrend is expected. On the other hand, if the long-term average is above a shorter-term average then a downtrend might be the expected outcome

look at the angle of the moving average
If it is mostly moving horizontally for an extended amount of time, then the price isn't trending, it is ranging. A trading range occurs when a security trades between consistent high and low prices for a period of time. 
If the moving average line is angled up, an uptrend is underway

Two popular trading patterns that use simple moving averages
1. A death cross occurs when the 50-day SMA crosses below the 200-day SMA. This is considered a bearish signal, that further losses are in store. 
2. The golden cross occurs when a short-term SMA breaks above a long-term SMA. Reinforced by high trading volumes, this can signal further gains are in store. 

More Patterns
Rectange https://www.investopedia.com/terms/r/rectangle.asp
Head And Shoulders Pattern https://www.investopedia.com/terms/h/head-shoulders.asp
Kicker Pattern https://www.investopedia.com/terms/k/kickerpattern.asp
Bullish Engulfing Pattern https://www.investopedia.com/terms/b/bullishengulfingpattern.asp
Triple Top https://www.investopedia.com/terms/t/tripletop.asp

Common simple moving averages
1. day trading: 5-, 8- and 13-bar simple moving averages 
2. short term: 10-day, 20-day
3. investors and long-term trend followers: 200-day, 100-day, and 50-day simple moving average 

uptrend describes the price movement of a financial asset when the overall direction is upward. Uptrends are characterized by higher peaks and troughs over time and imply bullish sentiment among investors
https://www.investopedia.com/terms/u/uptrend.asp

downtrend refers to the price action of a security that moves lower in price as it fluctuates over time. Downtrends are characterized by lower peaks and troughs and imply fundamental changes in the beliefs of investors.
https://www.investopedia.com/terms/d/downtrend.asp



'''

# https://pypi.org/project/yfinance/
# pip install yfinance



'''
EMA gives a higher weighting to recent prices, while the SMA assigns an equal weighting to all values.
Since EMAs place a higher weighting on recent data than on older data, they are more reactive to the latest price changes. Thats why the EMA is the preferred average among many traders

Common exponential moving averages 
1. short-term: 12-day and 26-day
2. long term: 50-day and 200-day
'''




import numpy as np
import pandas as pd
import yfinance as yf

# calling Yahoo finance API and requesting to get data for the last 1 week, with an interval of 90 minutes
# data = yf.download(tickers='MSFT', period = '1wk', interval = '90m')
data = yf.download("WMT", start="2020-05-01", end="2021-04-30")

ma = [10,20] #[5,8,13, 10,20, 50,100,200]
colors = ['orange', 'yellow']
for m in ma:
    data[str(m)+'d']=data['Close'].rolling(m).mean()

data  = data[['Close',str(ma[0])+'d',str(ma[1])+'d']]
data['flag']=np.where((data[str(ma[0])+'d']-data[str(ma[1])+'d'])<0, True, False) # short minus long ma

data['flag2']=['hold']*data.shape[0]

for r in range(1,data.shape[0]):
    if data['flag'][r-1]==True and data['flag'][r]==False: # s ma line is above
        data['flag2'][r]='buy'
    if data['flag'][r-1]==False and data['flag'][r]==True: # s ma line is above
        data['flag2'][r]='sell'

# When the shorter-term MA crosses above the longer-term MA, it's a buy signal, as it indicates that the trend is shifting up. This is known as a "golden cross." 
# when the shorter-term MA crosses below the longer-term MA, it's a sell signal, as it indicates that the trend is shifting down. This is known as a "dead/death cross." 


#from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = go.Figure()
#Candlestick
#fig.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close, name='mkt data'))
fig.add_trace(go.Line(x=data.index, y=data.Close, name='close $', line=dict(color='white', width=1))) # dot
for m, c in zip(ma, colors):
    fig.add_trace(go.Line(x=data.index, y=data['Close'].rolling(m).mean(), name=str(m)+'d', line=dict(color=c, width=1, dash='dot'))) # dash
fig.add_trace(go.Scatter(x=data[data.flag2!='hold'].index, y=list(data[data.flag2!='hold'].Close), name='signals', mode='markers', marker=dict(color='white')))

for xx, yy, tt in zip (data[data.flag2!='hold'].index, list(data[data.flag2!='hold'].Close+5), list(data[data.flag2!='hold'].flag2)):
    fig.add_annotation(dict(xref='x',yref='y',x=xx, y=yy, xanchor='center',yanchor='top',font=dict(family='Arial', size=10, color='white'),showarrow=False,text=tt))

fig.update_layout(template="plotly_dark", autosize=False, width=900, height=800, margin=dict(l=50, r=50, b=100, t=100, pad=4))
fig.show()


