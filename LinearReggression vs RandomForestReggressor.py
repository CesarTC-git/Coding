#!/usr/bin/env python
# coding: utf-8

## Create a model that predicts House values
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reading the csv file
housing_train = pd.read_csv("./train.csv")

# Extract data info
housing_train.info()

# Check firsts rows of data
pd.set_option('display.max_columns', None)
housing_train.head()

# Take a look at the different attributes
housing_train.hist(bins=50, figsize=(20,15))
plt.show()

# Check the correlation between attributes
corr_matrix = housing_train.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)

# Plot the correlation between attributes using Seaborn
import seaborn as sbn
plt.figure(figsize = (20,10))
sbn.heatmap(housing_train.corr().abs(), annot = True)

# Plot correlation between specific variables with Seaborn
housing_train_selec = housing_train.loc[:,['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]
plt.figure(figsize = (40,20))
sbn.heatmap(housing_train_selec.corr().abs(), annot = True)

from pandas.plotting import scatter_matrix
attributes = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF']
scatter_matrix(housing_train[attributes], figsize=(12,8))

# Prepare TRAIN and TEST datasets using only the specified attributes
from sklearn.model_selection import train_test_split
X = housing_train[attributes]
y = housing_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_test

# Prepare the Linear and the RandomForest reggressors
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
from sklearn.ensemble import RandomForestRegressor
rm = RandomForestRegressor()

lm_fit = lm.fit(X_train,y_train)
rm_fit = rm.fit(X_train,y_train)

# Obtaining predictions of each model for the TEST dataset
lm_predicciones = lm.predict(X_test)
rm_predicciones = rm.predict(X_test)

# Creating Dataframes for each prediction
lm_DTpredicciones = pd.DataFrame(lm_predicciones)
lm_DTpredicciones.reset_index(drop = True, inplace = True)
rm_DTpredicciones = pd.DataFrame(rm_predicciones)
rm_DTpredicciones.reset_index(drop = True, inplace = True)
DTy = pd.DataFrame(y_test)
DTy.reset_index(drop = True, inplace = True)
DTy.join(lm_DTpredicciones)


## Métricas
# Checking the Linear Reggression model returns reasonable values
lm_DTpredicciones
plt.plot(rm_DTpredicciones)

# Print main metrics using Scikit-Learn
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,rm_predicciones))
print('MSE:', metrics.mean_squared_error(y_test,rm_predicciones))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,rm_predicciones)))
from sklearn.metrics import mean_squared_log_error
print('Log RMSE:', metrics.mean_squared_log_error(y_test,rm_predicciones))

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,lm_predicciones))
print('MSE:', metrics.mean_squared_error(y_test,lm_predicciones))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,lm_predicciones)))
