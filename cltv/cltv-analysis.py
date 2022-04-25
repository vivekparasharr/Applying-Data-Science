# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:06:55 2020

@author: vivek
"""

# For computing CLTV we need historical data of customers but you will unable to calculate for new customers. To solve this problem Business Analyst develops machine learning models to predict the CLTV of newly customers. 


# step 1 - You can compute it by adding profit/revenue from customers in a given cycle. For Example, If the customer is associated with you for the last 3 years, you can sum all the profit in this 3 years. You can average the profit yearly or half-yearly or monthly, but in this approach, you cannot able to build a predictive model for new customers.
# step 2 - Build a regression model for existing customers. Take recent six-month data as independent variables and total revenue over three years as a dependent variable and build a regression model on this data.
# step 3 - Using the following equation: CLTV = ((Average Order Value x Purchase Frequency)/Churn Rate) x Profit margin.
            # Customer Value = Average Order Value * Purchase Frequency
            # Average Order Value = Total Revenue / Total Number of Orders
            # Purchase Frequency =  Total Number of Orders / Total Number of Customers
            # Churn Rate is the percentage of customers who have not ordered again
            # Customer Lifetime=1/Churn Rate
            # Repeat rate can be defined as the ratio of the number of customers with more than one order to the number of unique customers
            # Churn Rate= 1-Repeat Rate
                                               
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
import numpy as np



data = pd.read_excel("Data/Online_Retail.xlsx")
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
uk_data=uk_data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]

#Calulate total purchase
uk_data['TotalPurchase'] = uk_data['Quantity'] * uk_data['UnitPrice']


uk_data_group=uk_data.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'Quantity': lambda quant: quant.sum(),
                                        'TotalPurchase': lambda price: price.sum()})                                               

# Change the name of columns
uk_data_group.columns=['num_days','num_transactions','num_units','spent_money']
uk_data_group.head()

# Calculate CLTV formula variables
uk_data_group['avg_order_value']=uk_data_group['spent_money']/uk_data_group['num_transactions']
purchase_frequency=sum(uk_data_group['num_transactions'])/uk_data_group.shape[0]
repeat_rate=uk_data_group[uk_data_group.num_transactions > 1].shape[0]/uk_data_group.shape[0]
churn_rate=1-repeat_rate

# Profit Margin - Let's assume our business has approx 5% profit on the total sale.
uk_data_group['profit_margin']=uk_data_group['spent_money']*0.05

# Customer Value
uk_data_group['CLV']=(uk_data_group['avg_order_value']*purchase_frequency)/churn_rate

#Customer Lifetime Value
uk_data_group['cust_lifetime_value']=uk_data_group['CLV']*uk_data_group['profit_margin']


# Prediction Model for CLTV using Linear Regression

# First, Extract month and year from InvoiceDate.
uk_data['month_yr'] = uk_data['InvoiceDate'].apply(lambda x: x.strftime('%b-%Y'))

uk_data.head()

# pivot table takes the columns as input, and groups the entries into a two-dimensional table in such a way that provides a multidimensional summarization of the data
sale=uk_data.pivot_table(index=['CustomerID'],columns=['month_yr'],values='TotalPurchase',aggfunc='sum',fill_value=0).reset_index()


sale.head()
sale['CLV']=sale.iloc[:,2:].sum(axis=1)

# Defining dependent and independent (CLV) variables for the model
X=sale[['Dec-2011','Nov-2011', 'Oct-2011','Sep-2011','Aug-2011','Jul-2011']]
y=sale[['CLV']]


#split training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)


# import model
from sklearn.linear_model import LinearRegression
# instantiate
linreg = LinearRegression()
# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
# make predictions on the testing set
y_pred = linreg.predict(X_test)


# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# How Well Does the Model Fit the data?
from sklearn import metrics

# compute the R Square for model
print("R-Square:",metrics.r2_score(y_test, y_pred))

# This model has a higher R-squared (0.96). This model provides a better fit to the data.


# calculate MAE using scikit-learn
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))

#calculate mean squared error
print("MSE",metrics.mean_squared_error(y_test, y_pred))
# compute the RMSE of our predictions
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# RMSE is more popular than MSE and MAE because RMSE is interpretable with y because of the same units.






































