# the dataset was imported from Kaggle and it is the NASA JPL Asteroid Dataset
# it's a pretty huge dataset for Asteroids (it's the biggest one i worked with so far)
# The dataset contains various features related to asteroids, such as their size, orbital parameters, and other characteristics
# GOAL: classifying asteroids into different classes which will help us to identify Outer Main Belt (OMBs)
# and Mars-crossing Asteroids (MCAs), Main Belt Asteroids (MBAs) etc..



from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


asteroidataset = pd.read_csv('./datasets/asteroid.csv', low_memory=False)

# the target variable
y = asteroidataset['class'].values

# exploring the values of classes
unique_values = asteroidataset['class'].unique()
print(unique_values)


