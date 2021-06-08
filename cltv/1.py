
'''
Approaches to CLTV -
Historical Approach:
1. Aggregate Model  —  calculating the CLV by using the average revenue per customer based on past transactions. This method gives us a single value for the CLV.
2. Cohort Model  —  grouping the customers into different cohorts based on the transaction date, etc., and calculate the average revenue per cohort. This method gives CLV value for each cohort.

Predictive Approach:
3. Machine Learning Model  —  using regression techniques to fit on past data to predict the CLV.
4. Probabilistic Model  —  it tries to fit a probability distribution to the data and estimates the future count of transactions and monetary value for each transaction.
'''

'''
About the dataset -
This is a transactional data set that contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
Attribute Information:

InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter ‘c’, it indicates a cancellation.
StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invoice Date and time. Numeric, the day and time when each transaction was generated.
UnitPrice: Unit price. Numeric, Product price per unit in sterling.
CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal, the name of the country where each customer resides.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataprep.eda as dp

##############################################################

# importing the dataset
df = pd.read_excel('/Volumes/sandisk8gb/Documents/Code/Customer-Analytics/Data/Online_Retail.xlsx')
df.head(1)

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

# Calculating CLV for each cohort
months = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_CLV = []

df3.columns = ['Start_Month', 'Frequency', 'TotalSales'] # rename Age to Start_Month  

for i in range(1, 13):
    customer_m = df3[df3.Start_Month==i]
    
    Average_sales = round(np.mean(customer_m.TotalSales),2)
    
    Purchase_freq = round(np.mean(customer_m.Frequency), 2)
    
    Retention_rate = customer_m[customer_m.Frequency>1].shape[0]/customer_m.shape[0]
    churn = round(1 - Retention_rate, 2)
    
    CLV = round(((Average_sales * Purchase_freq/churn)) * Profit_margin, 2)
    
    Monthly_CLV.append(CLV)
