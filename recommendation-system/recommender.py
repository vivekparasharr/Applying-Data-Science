
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
    i = -1
    for r in df.columns:
        i+=1
        print(i, '|', r, ':', # column name
        df[r].unique().shape[0], # number of unique elements in the column
        '| example:', df[r][0]) # example of the first element in the column
vp_summ(bank_full)

X = bank_full.iloc[:,18:36].values

y = bank_full.iloc[:,17].values

LogReg = LogisticRegression()
LogReg.fit(X, y)

new_user = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
y_pred = LogReg.predict(new_user)
y_pred


