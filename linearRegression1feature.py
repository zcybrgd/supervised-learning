# Performing Supervised Learning: Single Variable Linear Regression
# Basic Algorithm: building a linear regression model to predict the salary of an employee based on their years of experience
# the dataset Salary_dataset was imported from Kaggle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
salary_per_years = pd.read_csv('./datasets/salary_dataset.csv', index_col=0)
# extract the independant variable (the feature)
X = salary_per_years['YearsExperience'].values
# extract the target
y = salary_per_years['Salary'].values
X = X.reshape(-1,1)
# instantiate the model
reg = LinearRegression()

# fit the training data
reg.fit(X,y)

# make predictions
predictions = reg.predict(X)

# visualize

# scatter the actual data points
plt.scatter(X, y, color='yellow')

# draw our predictions by a linear regression
plt.plot(X, predictions)

# adjusting the visualization process
plt.xlabel('Year of experience')
plt.ylabel('salary')
plt.title('Linear Regression model that predict the salary of an employee depending on their years of experience')
plt.show()

# performing cross validation
rsq = reg.score(X, y)
print("R^2: {}".format(rsq))

kf = KFold(n_splits=3, shuffle=True, random_state=2)

cv_scores = cross_val_score(reg, X, y, cv=kf)
print(cv_scores)