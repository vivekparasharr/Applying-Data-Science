
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_selection import SelectPercentile, univariate_selection, RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, RepeatedStratifiedKFold,\
StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, precision_recall_curve, plot_precision_recall_curve, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
import category_encoders as ce
from scipy import stats

'''
Context
A fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3.
Data Description
7043 observations with 33 variables

CustomerID: A unique ID that identifies each customer.
Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.
Country: The country of the customer’s primary residence.
State: The state of the customer’s primary residence.
City: The city of the customer’s primary residence.
Zip Code: The zip code of the customer’s primary residence.
Lat Long: The combined latitude and longitude of the customer’s primary residence.
Latitude: The latitude of the customer’s primary residence.
Longitude: The longitude of the customer’s primary residence.
Gender: The customer’s gender: Male, Female
Senior Citizen: Indicates if the customer is 65 or older: Yes, No
Partner: Indicate if the customer has a partner: Yes, No
Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
Tenure Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
Device Protection: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
Churn Label: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.
Churn Value: 1 = the customer left the company this quarter. 0 = the customer remained with the company. Directly related to Churn Label.
Churn Score: A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
CLTV: Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.
Churn Reason: A customer’s specific reason for leaving the company. Directly related to Churn Category.
'''
# https://www.kaggle.com/yeanzc/telco-customer-churn-ibm-dataset
# https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2018/09/12/base-samples-for-ibm-cognos-analytics
df = pd.read_excel('Data/Telco_customer_churn.xlsx')

df.head(3)
df.info()

'''
Problem Statement
In a telco company, there are promotional costs known as Acquisition Cost and Retention Cost. Acquisition Cost is the cost for a company to acquire new customers. Meanwhile, Retention Cost is the cost for the company to retain existing customers.
Due to human limitations, we are often wrong to predict which customers will churn and which customers will retain. So that the allocation of funds can be wrong so that the funds issued become larger.
Moreover, according to some sources, the acquisition cost is 5x greater than the retantion cost. If we are wrong in predicting a customer who will actually churn, but it turns out that we predict as a customer who will retain, then we need to spend more than it should be.

What to do
Try to create a Machine Learning model to predict customer churn and retantion.

Goal
Machine Learning has a goal so that cost allocation can be done as precisely as possible.

Value
There is no wasted cost allocation.
'''

df.head(3)
df.info()
df.describe()



def report(df):
    col = []
    d_type = []
    uniques = []
    n_uniques = []
    for i in df.columns:
        col.append(i)
        d_type.append(df[i].dtypes)
        uniques.append(df[i].unique()[:5])
        n_uniques.append(df[i].nunique())
    
    return pd.DataFrame({'Column': col, 'd_type': d_type, 'unique_sample': uniques, 'n_uniques': n_uniques})

report(df)

# we review the data and see which columns might not add any value
# there are several columns that have one unique value, namely the column [Count, Country, State]. 
# In addition, I will not use the CustomerID column because the customerID does not determine the probability that someone will churn or not.
# Zip code, Lat Long, Latitude, Longitude will also be deleted. I won't use it to build Machine Learning.

# one time run
new_col = df.columns.str.replace(' ', '_')
df.columns = new_col

df.drop('CustomerID Count City Country State Zip_Code Lat_Long Latitude Longitude'.split(), axis=1, inplace=True)


# EDA
df.groupby('Churn_Label').count()['CustomerID'].plot(kind='pie', autopct='%.2f%%')

# the proportion for male and female is almost the same
df.groupby('Gender').count()['CustomerID'].plot(kind='pie', autopct='%.2f%%')

