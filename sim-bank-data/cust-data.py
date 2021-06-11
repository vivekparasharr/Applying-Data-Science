

from sklearn.datasets import make_classification
features, output = make_classification(n_samples = 50,
                                       n_features = 5,
                                       n_informative = 5,
                                       n_redundant = 0,
                                       n_classes = 3,
                                       weights = [.2, .3, .8])


# Step 1: Create a DataFrame
df = pd.DataFrame({'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]})

# Step 2: Create a Database
import sqlite3
conn = sqlite3.connect('/Users/vivekparashar/Desktop/TestDB1.db')
c = conn.cursor()

# create the 'CARS' table
c.execute('CREATE TABLE CARS (Brand text, Price number)')
conn.commit()

# Step 3: Save from Pandas DataFrame to SQL
df.to_sql('CARS', conn, if_exists='replace', index = False)

# Access data from SQL database to see if it worked
# Recurse over data using for loop
c.execute('''  
SELECT * FROM CARS
          ''')
for row in c.fetchall():
    print (row)

# Pull  data into a dataframe
c.execute('''  
SELECT * FROM CARS
          ''')
df2 = pd.DataFrame(c.fetchall(), columns=['Brand','Price'])    
print(df2)

# Drop the table
c.execute('DROP TABLE CARS') # drop table

c.close() # close connection



