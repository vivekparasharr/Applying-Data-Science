
'''
Approaches to CLTV -
Historical Approach:
1. Aggregate Model  —  calculating the CLV by using the average revenue per customer based on past transactions. This method gives us a single value for the CLV.
2. Cohort Model  —  grouping the customers into different cohorts based on the transaction date, etc., and calculate the average revenue per cohort. This method gives CLV value for each cohort.

Predictive Approach:
3. Machine Learning Model  —  using regression techniques to fit on past data to predict the CLV.
4. Probabilistic Model  —  it tries to fit a probability distribution to the data and estimates the future count of transactions and monetary value for each transaction.
'''

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataprep.eda as dp

##############################################################
######   move czech bank (berka) data to a sql database ######
##############################################################

# importing the dataset
client = pd.read_csv('/Volumes/sandisk8gb/Documents/Code/Customer-Analytics/Data/czech-bank-berka/data/client.csv')

# create a database
conn = sqlite3.connect('/Volumes/sandisk8gb/Documents/Code/Customer-Analytics/Data/czech-bank-berka/data/czech-bank-berka.db')
c = conn.cursor()

# create an empty table in db to save the df
c.execute('CREATE TABLE client (client_id number, birth_number number, district_id number)') # column data type can be text or number
conn.commit()

# save from pandas df to sql db
client.to_sql('client', conn, if_exists='replace', index = False)  

# access data from SQL database (recurse over data using for loop)
c.execute('''  SELECT * FROM client where client_id<10 ''')
for row in c.fetchall():
    print (row)

# Pull  data into a dataframe
c.execute('''  SELECT client_id, birth_number FROM client ''')
df2 = pd.DataFrame(c.fetchall(), columns=['client_id','birth_number'])    

# drop the table
c.execute('DROP TABLE client') 

# close connection
c.close() 

##############################################################

conn = sqlite3.connect('/Volumes/sandisk8gb/Documents/Code/Customer-Analytics/Data/td-bank/td-bank.db')
c = conn.cursor()
c.execute('''  SELECT * FROM rfm ''')
df = pd.DataFrame(c.fetchall(), columns=['customerid','frequency', 'recency', 'monetry_value', 'T'])    

# aggregate model
# Calculating the necessary variables for CLV calculation
df['TotalSales'] = df.frequency * df.monetry_value

Average_sales = round(np.mean(df.TotalSales),2)
print(f"Average sales: ${Average_sales}")

Purchase_freq = round(np.mean(df.frequency), 2)
print(f"Purchase Frequency: {Purchase_freq}")

Retention_rate = df[df.frequency>1].shape[0]/df.shape[0]
churn = round(1 - Retention_rate, 2)
print(f"Churn: {churn}%")

# assuming the Profit margin for each transaction to be roughly 5%
Profit_margin = 0.05

# Calculating the CLV
CLV = round(((Average_sales * Purchase_freq/churn)) * Profit_margin, 2)
print(f"The Customer Lifetime Value (CLV) for each customer is: ${CLV}")

##############################################################

# Cohort Model
# Instead of assuming all customers to be one group, we split them into multiple groups and calculate the CLV for each group
# most common way to group customers into cohorts is by the start date of a customer, typically by month
# best choice will depend on the customer acquisition rate, seasonality of business, and whether additional customer information can be used

# Transforming the data to customer level for the analysis
df = df.groupby('CustomerID').agg(
    {'InvoiceDate':lambda x: x.min().month, 
    'InvoiceNo': lambda x: len(x),
    'TotalSales': lambda x: sum(x)}
    )
df3.columns = ['Start_Month', 'Frequency', 'TotalSales'] # rename Age to Start_Month  

# handle division by 0
def weird_division(n, d):
    return n / d if d else 0

# Calculating CLV for each cohort
months = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_CLV = []

for i in range(1, 13):
    customer_m = df3[df3.Start_Month==i]
    
    Average_sales = round(np.mean(customer_m.TotalSales),2)
    
    Purchase_freq = round(np.mean(customer_m.Frequency), 2)
    
    Retention_rate = customer_m[customer_m.Frequency>1].shape[0]/customer_m.shape[0]
    churn = round(1 - Retention_rate, 2)
    
    CLV = round(((Average_sales * weird_division(Purchase_freq, churn))) * Profit_margin, 2)
    
    Monthly_CLV.append(CLV)

# review the output
monthly_clv = pd.DataFrame(zip(months, Monthly_CLV), columns=['Months', 'CLV'])
display(monthly_clv.style.background_gradient())

'''
we have 12 different CLV value for 12 months from Jan-Dec. And it is pretty clear that, customers who are acquired in different months have different CLV values attached to them. This is because, they could be acquired using different campaigns etc., so thier behaviour might be different from others
'''

##############################################################




# cleaning the dataset
# select features we need - CustomerID, InvoiceDate, Quantity and Total Sales (Quantity * UnitPrice)
df2 = df[['Quantity', 'InvoiceNo', 'InvoiceDate', 'UnitPrice', 'CustomerID']]
df2['TotalSales'] = df2.Quantity * df2.UnitPrice
df2.shape

# review descriptive statistics
df2.describe()

# drop negative sales due to returns
df3=df2[df2.TotalSales>0] 
df3.shape

# check how many CustomerID's are missing
dp.plot_missing(df2, 'CustomerID') 
pd.DataFrame(zip(df2.isnull().sum(), df2.isnull().sum()/len(df2)), columns=['Count', 'Proportion'], index=df2.columns) # alternate approach

# drop rows with null CustomerID
df2 = df2[pd.notnull(df2.CustomerID)] 

##############################################################

# aggregate model
# assumes a constant average spend and churn rate for all the customers, and produces a single value for CLV at an overall Level
# downside - unrealistic estimates if some of the customers transacted in high value and high volume

'''
CLV = ((Average Sales X Purchase Frequency) / Churn) X Profit Margin
Where,
Average Sales = TotalSales/Total no. of orders
Purchase Frequency = Total no. of orders/Total unique customers
Retention rate = Total no. of orders greater than 1/ Total unique customers
Churn = 1 - Retention rate
Profit Margin = Based on business context
'''

# Transforming the data to customer level for the analysis
df3 = df2.groupby('CustomerID').agg(
    {'InvoiceDate':lambda x: (x.max() - x.min()).days,
    'InvoiceNo': lambda x: len(x),
    'TotalSales': lambda x: sum(x)}
    )
df3.columns = ['Age', 'Frequency', 'TotalSales']
df3.head(2)

# Calculating the necessary variables for CLV calculation
Average_sales = round(np.mean(df3.TotalSales),2)
print(f"Average sales: ${Average_sales}")

Purchase_freq = round(np.mean(df3.Frequency), 2)
print(f"Purchase Frequency: {Purchase_freq}")

Retention_rate = df3[df3.Frequency>1].shape[0]/df3.shape[0]
churn = round(1 - Retention_rate, 2)
print(f"Churn: {churn}%")

# assuming the Profit margin for each transaction to be roughly 5%
Profit_margin = 0.05

# Calculating the CLV
CLV = round(((Average_sales * Purchase_freq/churn)) * Profit_margin, 2)
print(f"The Customer Lifetime Value (CLV) for each customer is: ${CLV}")

##############################################################

# Cohort Model
# Instead of assuming all customers to be one group, we split them into multiple groups and calculate the CLV for each group
# most common way to group customers into cohorts is by the start date of a customer, typically by month
# best choice will depend on the customer acquisition rate, seasonality of business, and whether additional customer information can be used

# Transforming the data to customer level for the analysis
df3 = df2.groupby('CustomerID').agg(
    {'InvoiceDate':lambda x: x.min().month, 
    'InvoiceNo': lambda x: len(x),
    'TotalSales': lambda x: sum(x)}
    )
df3.columns = ['Start_Month', 'Frequency', 'TotalSales'] # rename Age to Start_Month  

# handle division by 0
def weird_division(n, d):
    return n / d if d else 0

# Calculating CLV for each cohort
months = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_CLV = []

for i in range(1, 13):
    customer_m = df3[df3.Start_Month==i]
    
    Average_sales = round(np.mean(customer_m.TotalSales),2)
    
    Purchase_freq = round(np.mean(customer_m.Frequency), 2)
    
    Retention_rate = customer_m[customer_m.Frequency>1].shape[0]/customer_m.shape[0]
    churn = round(1 - Retention_rate, 2)
    
    CLV = round(((Average_sales * weird_division(Purchase_freq, churn))) * Profit_margin, 2)
    
    Monthly_CLV.append(CLV)

# review the output
monthly_clv = pd.DataFrame(zip(months, Monthly_CLV), columns=['Months', 'CLV'])
display(monthly_clv.style.background_gradient())

'''
we have 12 different CLV value for 12 months from Jan-Dec. And it is pretty clear that, customers who are acquired in different months have different CLV values attached to them. This is because, they could be acquired using different campaigns etc., so thier behaviour might be different from others
'''

##############################################################

# Pareto/NBD model
# Pareto/NBD model tries to predict the future transactions of each customer


##############################################################

# BG/NBD Model (with Gamma-Gamma extension)
'''
BG/NBD stands for Beta Geometric/Negative Binomial Distribution

BG/NBD model tries to predict the future transactions of each customer
It is then combined with Gamma-Gamma model, which adds the monetary aspect of the customer transaction 
and we finally get the customer lifetime value (CLV)

The BG/NBD model has few assumptions: These are some of the assumptions this model considers for predicting the future transactions of a customer.
    When a user is active, number of transactions in a time t is described by Poisson distribution with rate lambda.
    Heterogeneity in transaction across users (difference in purchasing behavior across users) has Gamma distribution with shape parameter r and scale parameter a.
    Users may become inactive after any transaction with probability p and their dropout point is distributed between purchases with Geometric distribution.
    Heterogeneity in dropout probability has Beta distribution with the two shape parameters alpha and beta.
    Transaction rate and dropout probability vary independently across users.
'''

# Importing the lifetimes package
import lifetimes

# First we need to create a summary table from the transactions data. The summary table is nothing but RFM table. (RFM - Recency, Frequency and Monetary value)
'''
    frequency - the number of repeat purchases (more than 1 purchases)
    recency - the time between the first and the last transaction
    T - the time between the first purchase and the end of the transaction period
    monetary_value - it is the mean of a given customers sales value
NOTE: If you closely look at the definition of recency and T, you can find that, the actual value of recency should be (T - recency), because the definition of recency is how recent a customer made a transaction with the business.
'''
# Creating the summary data using summary_data_from_transaction_data function
summary = lifetimes.utils.summary_data_from_transaction_data(df2, 'CustomerID', 'InvoiceDate', 'TotalSales' )
summary = summary.reset_index()
summary.head()

'''SQL statement to transform transactional data into RFM data
SELECT
  customer_id,
  COUNT(distinct date(transaction_at)) - 1 as frequency,
  datediff('day', MIN(transaction_at), MAX(transaction_at)) as recency,
  AVG(total_price) as monetary_value,
  datediff('day', CURRENT_DATE, MIN(transaction_at)) as T
FROM orders
GROUP BY customer_id
'''

# value of 0 in frequency and recency means that, these are one time buyers
# Let's check how many such one time buyers are there in our data
# Create a distribution of frequency to understand the customer frequence level
summary['frequency'].plot(kind='hist', bins=50)

# Fitting the BG/NBD model
# Beta Geo Fitter, also known as BG/NBD model.
bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Model summary table shows the estimated distribution parameter values from the historical data
bgf.summary

# predict with model
# use the trained model to predict the future transactions and the customer churn rate

# Compute the customer alive probability
# model.conditional_probability_alive(): This method computes the probability that a customer with history (frequency, recency, T) is currently alive
summary['probability_alive'] = bgf.conditional_probability_alive(summary['frequency'], summary['recency'], summary['T'])
summary.head(10)

# Visual representation of relationship between recency and frequency
# plot_probabilty_alive_matrix(model): This function from lifetimes.plotting will help to visually analyze the relationship between recency & frequency and the customer being alive
from lifetimes.plotting import plot_probability_alive_matrix
fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)

# predict the likely future transactions for each customer
# model.conditional_expected_number_of_purchases_up_to_time(): Calculate the expected number of repeat purchases up to time t for a randomly chosen individual from the population (or the whole population), given they have purchase history (frequency, recency, T)
# Predict future transaction for the next 30 days based on historical dataa
t = 30
summary['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T']),2)
summary.sort_values(by='pred_num_txn', ascending=False).head(10).reset_index()


# Gamma-Gamma Model
# lifetimes.fitters.gamma_gamma_fitter module also known as Gamma-Gamma Model
# Now that we predicted the expected future transactions, we now need to predict the future monetary value of each transactions.
'''
Some of the key assumptions of Gamma-Gamma model are:
    The monetary value of a customer's given transaction varies randomly around their average transaction value.
    Average transaction value varies across customers but do not vary over time for any given customer.
    The distribution of average transaction values across customers is independent of the transaction process.
'''
# We are considering only customers who made repeat purchases with the business i.e., frequency > 0. Because, if frequency is 0, it means that they are one time customer and are considered already dead.
# final assumption (no relationship between frequency and monetary value of transactions) can be validated using Pearson correlation.

# Checking the relationship between frequency and monetary_value
# return_customers_summary = summary[summary['frequency']>0]
return_customers_summary = summary[summary['frequency']>0][summary['monetary_value']>0] # added additional filter to exclude transactions with <=0 monetary_value
print(return_customers_summary.shape)
return_customers_summary.head()

# Checking the relationship between frequency and monetary_value
return_customers_summary[['frequency', 'monetary_value']].corr()

# Modeling the monetary value using Gamma-Gamma Model
ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(return_customers_summary['frequency'],
       return_customers_summary['monetary_value'])

# Summary of the fitted parameters
ggf.summary


# predict using the model
# predict the expected average profit for each each transaction and Customer Lifetime Value using the model

# Calculating the conditional expected average profit for each customer per transaction
# model.conditional_expected_average_profit(): This method computes the conditional expectation of the average profit per transaction for a group of one or more customers
summary = summary[summary['monetary_value'] >0]
summary['exp_avg_sales'] = ggf.conditional_expected_average_profit(summary['frequency'],
                                       summary['monetary_value'])
summary.head()
# Checking the expected average value and the actual average value in the data to make sure the values are good
print(f"Expected Average Sales: {summary['exp_avg_sales'].mean()}")
print(f"Actual Average Sales: {summary['monetary_value'].mean()}")

# calculate the customer lifetime value directly using the method from the lifetimes package
# model.customer_lifetime_value(): This method computes the average lifetime value of a group of one or more customers. This method takes in BG/NBD model and the prediction horizon as a parameter to calculate the CLV
'''
Three main important thing to note here is:
1. time: This parameter in customer_lifetime_value() method takes in terms of months i.e., t=1 means one month and so on.
2. freq: This parameter is where you will specify the time unit your data is in. If your data is in daily level then "D", monthly "M" and so on.
3. discount_rate: This parameter is based on the concept of DCF (discounted cash flow), where you will discount the future monetary value by a discount rate to get the present value of that cash flow. In the documentation, it is given that for monthly it is 0.01 (annually ~12.7%)
'''
# Predicting Customer Lifetime Value for the next 30 days
summary['predicted_clv'] =      ggf.customer_lifetime_value(bgf,
                                                               summary['frequency'],
                                                               summary['recency'],
                                                               summary['T'],
                                                               summary['monetary_value'],
                                                               time=1,     # lifetime in months
                                                               freq='D',   # frequency in which the data is present(T)      
                                                               discount_rate=0.01) # discount rate
summary.head()

# can also calculate the CLV manually from the predicted number of future transactions (pred_num_txn) and expected average sales per transaction (exp_avg_sales)
summary['manual_predicted_clv'] = summary['pred_num_txn'] * summary['exp_avg_sales']
summary.head()

# CLV in terms of profit (profit margin is 5%)
# One thing to note here is that, both the values we have calculated for CLV is the sales value, not the actual profit. To get the net profit for each customer, we can either create profit value in the begining by multiplying sales value with profit margin or we can do that now.
profit_margin = 0.05
summary['CLV'] = summary['predicted_clv'] * profit_margin
summary.head()

# Distribution of CLV for the business in the next 30 days
summary['CLV'].describe()
summary['CLV'].describe().plot(kind='hist', bins=50)

