
'''
I broke the alert system process into four pieces:
Retrieve the crypto’s current price (CoinAPI)
Retrieve the crypto’s historical price data (CoinAPI)
Determine if current price is a “bargain”
Summarize findings via push notification (Slack API)
'''

import requests
import pandas as pd
from scipy import stats
import time

crypto = 'BTC'
url = 'https://rest.coinapi.io/v1/exchangerate/{0}/USD'.format(crypto)
headers = {'X-CoinAPI-Key' : 'YOUR-KEY-HERE'}
response = requests.get(url, headers = headers)

content = response.json()
current_price = content['rate']
current_time = content['time']


# get historical prices (30 days)
crypto = 'BTC'
url = 'https://rest.coinapi.io/v1/ohlcv/{0}/USD/latest?period_id=1DAY&limit=30'.format(crypto)
headers = {'X-CoinAPI-Key' : 'YOUR-KEY-HERE'}
response = requests.get(url, headers=headers)

content = response.json()
df_30 = pd.DataFrame(content)


#If the current price is less than the 20% percentile of prices from the last 30 days, it’s considered a bargain. If it’s greater than the 80% percentile, it’s a “rip-off”. 

day_30_percentile = stats.percentileofscore(df_30.price_close, current_price)

# determine 
if day_30_percentile <= 20:
    status = 'BARGIN'
elif day_30_percentile <= 80:
    status = 'NORMAL'
else:
    status = 'RIP-OFF'

slack_api_url = 'https://slack.com/api/chat.postMessage'
message = '{0} is a {1} today.format(crypto, status, current_price_formatted, percentile_formatted)
data = {'token': slack_token, channel": "CL3N5F9V5", "text": text}
r = requests.post(url = slack_api_url, data = data)

  


import requests
import pandas as pd

def get_crypto_price(symbol, exchange, start_date = None):
    api_key = 'YOUR API KEY'
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    df = df.rename(columns = {'1a. open (USD)': 'open', '2a. high (USD)': 'high', '3a. low (USD)': 'low', '4a. close (USD)': 'close', '5. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    return df

shib = get_crypto_price(symbol = 'SHIB', exchange = 'CAD', start_date = '2020-01-01')
shib

# Raw Package
import numpy as np
import pandas as pd
#Data Source
import yfinance as yf
#Data viz
import plotly.graph_objs as go
data = yf.download(tickers='SHIB-CAD', period = '22h', interval = '1m')

do (
  
)




