# Regularizing our Model with Ridge and Lasso to Prevent Overfitting
# Predicting Fuel Efficiency (mpg) of Automobiles using Multiple Features

import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
carsMpg = pd.read_csv('./datasets/reg.csv')

# Cleaning data

# Replace '?' with NaN in the 'horsepower' column
carsMpg['horsepower'] = pd.to_numeric(carsMpg['horsepower'], errors='coerce')

# Drop rows with NaN values in the 'horsepower' column

carsMpg.dropna(subset=['horsepower'], inplace=True)


carsMpg['horsepower'] = carsMpg['horsepower'].astype(float)

# extract the data from the df
X = carsMpg.drop(['mpg','name'], axis=1).values
y = carsMpg['mpg'].values

# test on 40% of the data, the rest is trained
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=28)

# Regression Regularization

# Ridge
ridgeScores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    ridgeScores.append(ridge.score(X_test, y_test))

print(ridgeScores)

# Lasso

# i am not going to repeat the same code purpose that i used for Ridge, instead we're going to use Lasso for feature selection
features = carsMpg.drop(['mpg','name'], axis=1).columns
lasso = Lasso(alpha=0.3)
lasso.fit(X, y)
# lasso coef refers to the coefficients or weights assigned to each feature in the linear regression model
lasso_coef = lasso.coef_
print(features)
plt.bar(features, lasso_coef)
plt.xticks(rotation=45)
plt.show()
