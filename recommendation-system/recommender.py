
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



