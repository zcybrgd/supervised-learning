# Performing Supervised Learning: Multiple Variable Linear Regression
# basic algorithm: Predicting Performance Index of students Based on many features related to their Study Habits and Activities
# + using model performance metrics such as R-squared and RMSE
# the dataset Student_performance was imported from Kaggle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data into a pandas dataframe
student_perform = pd.read_csv('./datasets/student_Performance.csv')

# Encode 'Extracurricular Activities' column to numerical values (0 for 'No' and 1 for 'Yes')
student_perform['Extracurricular Activities'] = student_perform['Extracurricular Activities'].map({'No': 0, 'Yes': 1})

# extract the data from the df
X = student_perform.drop('Performance Index', axis=1).values
y = student_perform['Performance Index'].values

# test on 35% of the data, the rest is trained
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=13)

# instantiate the model
reg = LinearRegression()

# fit the training data
reg.fit(X_train,y_train)

# make predictions
predictions = reg.predict(X_test)

# observing the model performance on some points
print("Predictions: {}, Actual Values: {}".format(predictions[:5], y_test[:5]))

# compute R-squared
rsq = reg.score(X_test, y_test)
# compute RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)

print("R^2: {}".format(rsq))
print("RMSE: {}".format(rmse))