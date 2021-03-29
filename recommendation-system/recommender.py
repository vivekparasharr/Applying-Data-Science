
# Popularity-Based Recommenders
import pandas as pd
import numpy as np

frame = pd.read_csv('01_02/rating_final.csv')
cuisine = pd.read_csv('01_02/chefmozcuisine.csv')

frame.head()
cuisine.head()

## Recommending based on counts
rating_count = frame.groupby('placeID').count()[['rating']]
most_rated_places = rating_count.sort_values('rating', ascending=False).head().reset_index()[['placeID']]
summary = pd.merge(most_rated_places, cuisine, on='placeID')

cuisine['Rcuisine'].describe()

##

import numpy as np
import pandas as pd


# Chapter 2 -  Machine Learning Based Recommendation Systems
## Segment 1 - Classification-based Collaborative Filtering Systems
## Logistic Regression as a Classifier

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression

bank_full = pd.read_csv('02_01/bank_full_w_dummy_vars.csv')
bank_full.head()

bank_full.info()

# summary of a database in terms of number of unique elements in each column
def vp_summ(df):
    print('#columns:', df.shape[1]) # number of columns
    print('#rows:', df.shape[0]) # number of rows
    for r in df.columns:
        print(r, ':', # column name
        df[r].unique().shape[0], # number of unique elements in the column
        '| example:', df[r][0]) # example of the first element in the column
vp_summ(bank_full)

X = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values

y = bank_full.ix[:,17].values




