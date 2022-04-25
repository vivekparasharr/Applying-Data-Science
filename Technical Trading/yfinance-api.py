
# https://pypi.org/project/yfinance/

import yfinance as yf
import pandas as pd
from pathlib import Path

msft = yf.Ticker("MSFT")

# get stock info
msft.info

# get historical market data
hist = msft.history(period="max")

##############################################################################################

# If you want to use a proxy server for downloading data, use:
msft.history(..., proxy="PROXY_SERVER")

##############################################################################################

# To initialize multiple Ticker objects, use
tickers = yf.Tickers('msft aapl goog')
# ^ returns a named tuple of Ticker objects
# access each ticker using (example)
tickers.tickers.MSFT.info
tickers.tickers.AAPL.history(period="1mo")
tickers.tickers.GOOG.actions

##############################################################################################

# Fetching data for multiple tickers
data = yf.download("SPY AAPL", start="2017-01-01", end="2017-04-30")

##############################################################################################

# Typical download string
data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "SPY AAPL MSFT",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "ytd",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1m",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )


# Download all tickers into single dataframe with single level column headers
# Option 1
tickerStrings = ['AAPL', 'MSFT']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='2d')
    data['ticker'] = ticker  # add this column becasue the dataframe doesn't contain a column with the ticker
    df_list.append(data)
df = pd.concat(df_list) # combine all dataframes into a single dataframe
# df.to_csv('ticker.csv') # save to csv

# Option 2
tickerStrings = ['AAPL', 'MSFT']
df = yf.download(tickerStrings, group_by='Ticker', period='2d')
df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

##############################################################################################

# Read yfinance csv already stored with multi-level column names
# If you wish to keep, and read in a file with a multi-level column index, use the following code, 
# which will return the dataframe to its original form.
df = pd.read_csv('test.csv', header=[0, 1])
df.drop([0], axis=0, inplace=True)  # drop this row because it only has one column with Date in it
df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] = pd.to_datetime(df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')], format='%Y-%m-%d')  # convert the first column to a datetime
df.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), inplace=True)  # set the first column as the index
df.index.name = None  # rename the index

##############################################################################################

# Flatten multi-level columns into a single level and add a ticker column
# If the ticker symbol is level=0 (top) of the column names
# When group_by='Ticker' is used
df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
# If the ticker symbol is level=1 (bottom) of the column names
df.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)


# Download each ticker and save it to a separate file
tickerStrings = ['AAPL', 'MSFT']
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period=prd, interval=intv)
    data['ticker'] = ticker  # add this column becasue the dataframe doesn't contain a column with the ticker
    data.to_csv(f'ticker_{ticker}.csv')  # ticker_AAPL.csv for example

##############################################################################################

# Read in multiple files saved with the previous section and create a single dataframe
# set the path to the files
p = Path('c:/path_to_files')
files = list(p.glob('ticker_*.csv')) # find the files
df_list = list() # read the files into a dataframe
for file in files:
    df_list.append(pd.read_csv(file))
df = pd.concat(df_list) # combine dataframes

##############################################################################################

# show actions (dividends, splits)
msft.actions

# show dividends
msft.dividends

# show splits
msft.splits

# show financials
msft.financials
msft.quarterly_financials

# show major holders
msft.major_holders

# show institutional holders
msft.institutional_holders

# show balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet

# show cashflow
msft.cashflow
msft.quarterly_cashflow

# show earnings
msft.earnings
msft.quarterly_earnings

# show sustainability
msft.sustainability

# show analysts recommendations
msft.recommendations

# show next event (earnings, etc)
msft.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options

# get option chain for specific expiration
opt = msft.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts

