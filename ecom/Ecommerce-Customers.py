
# https://www.kaggle.com/arpitsomani/ecommerce-customers-linear-regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/Ecommerce-Customers.csv')
df.head(3)
df.columns

'''
We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:

    Avg. Session Length: Average session of in-store style advice sessions.
    Time on App: Average time spent on App in minutes
    Time on Website: Average time spent on Website in minutes
    Length of Membership: How many years the customer has been a member.
'''

df.info()

df.describe()

sns.pairplot(df)

sns.distplot(df['Yearly Amount Spent'])

sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
sns.jointplot(x='Avg. Session Length',y='Yearly Amount Spent',data=df)

sns.heatmap(df.corr(),annot=True,cmap='Greens')



x=df[[ 'Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)



print(lm.intercept_)
lm.coef_

cdf=pd.DataFrame(lm.coef_,x.columns,columns=['coeff'])




# Getting prediction 
predictions=lm.predict(x_test)

plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('predicted values')




from sklearn import metrics


metrics.mean_absolute_error(y_test,predictions)

metrics.mean_squared_error(y_test,predictions)

np.sqrt(metrics.mean_squared_error(y_test,predictions))

metrics.explained_variance_score(y_test,predictions)

sns.distplot((y_test-predictions),bins=50)

cdf



