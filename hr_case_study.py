
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('Data/HR_Case_Study.csv', encoding = 'ISO-8859-1')

df.info()

df[['Name','Tenure in Company','Gender']].\
    pivot(index='Name', columns='Gender', values='Tenure in Company').plot(kind='hist')

df.Location.unique()

df.groupby('Location').count()[['Name']].plot(kind='bar', args=())
plt.ylabel('Number of Employees')
