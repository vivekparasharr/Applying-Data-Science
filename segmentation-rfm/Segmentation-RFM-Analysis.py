# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:31:21 2020

@author: vivek
"""

import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt

data = pd.read_excel("Online_Retail.xlsx")
data.head()

data.info()

# drop rows where customer id is null
data= data[pd.notnull(data['CustomerID'])]

# remove duplicate rows
# Only consider certain columns for identifying duplicates, by default use all of the columns.
filtered_data=data[['Country','CustomerID']].drop_duplicates()
filtered_data.info()

#Top ten country's customer
filtered_data.Country.value_counts()[:10].plot(kind='bar')


# since most customers are from uk, lets just focus on them
uk_data=data[data.Country=='United Kingdom']
uk_data.info()
uk_data.describe()

# we notice that some customers have ordered negative quantity, lets exclude those
uk_data = uk_data[(uk_data['Quantity']>0)]
uk_data.info()


# filter the necessary columns for RFM analysis
# only need her five columns CustomerID, InvoiceDate, InvoiceNo, Quantity, and UnitPrice
uk_data=uk_data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]
uk_data['TotalPrice'] = uk_data['Quantity'] * uk_data['UnitPrice']

uk_data['InvoiceDate'].min(),uk_data['InvoiceDate'].max()

# redefine present based on max invoice date
PRESENT = dt.datetime(2011,12,10)
uk_data['InvoiceDate'] = pd.to_datetime(uk_data['InvoiceDate'])

uk_data.head()

# RFM Analysis
# For Recency, Calculate the number of days between present date and date of last purchase each customer.
# For Frequency, Calculate the number of orders for each customer.
# For Monetary, Calculate sum of purchase price for each customer

rfm= uk_data.groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'TotalPrice': lambda price: price.sum()})

rfm.head(5)

# we need to change the column names because we used the actual col names for lambda functions

rfm.columns
rfm.columns=['recency','frequency','monetary']

rfm['recency'] = rfm['recency'].astype(int)

rfm.head()


# Computing Quantile of RFM values
# Customers with the lowest recency, highest frequency and monetary amounts considered as top customers.
# qcut() is Quantile-based discretization function. 
# qcut bins the data based on sample quantiles. For example, 1000 values for 4 
# quantiles would produce a categorical object indicating quantile membership for 
# each customer.

rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])


rfm.head()


# RFM Result Interpretation
# Combine all three quartiles(r_quartile,f_quartile,m_quartile) in a single column, 
# this rank will help you to segment the customers well group.

rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()


# Filter out Top/Best cusotmers
rfm[rfm['RFM_Score']=='111'].sort_values('monetary', ascending=False).head()























