
'''
Usually, about 80% of the time spent on a data science project is getting and cleaning data. Thanks to the quandl financial library, that was reduced to about 5% for this project. Quandl can be installed with pip from the command line, lets you access thousands of financial indicators with a single line of Python, and allows up to 50 requests a day without signing up. If you sign up for a free account, you get an api key that allows unlimited requests.

First, we import the required libraries and get some data. Quandl automatically puts our data into a pandas dataframe, the data structure of choice for data science. (For other companies, just replace the ‘TSLA’ or ‘GM’ with the stock ticker. You can also specify a date range).
'''

import quandl
import pandas as pd
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = 'getyourownkey!'# Retrieve TSLA data from Quandl
tesla = quandl.get('WIKI/TSLA')# Retrieve the GM data from Quandl
gm = quandl.get('WIKI/GM')

plt.plot(tesla.index, tesla['Adj. Close'], 'r')
plt.title('Tesla Stock Price')
plt.ylabel('Price ($)');
plt.show();

plt.plot(gm.index, gm['Adj. Close'])
plt.title('GM Stock Price')
plt.ylabel('Price ($)');
plt.show()

