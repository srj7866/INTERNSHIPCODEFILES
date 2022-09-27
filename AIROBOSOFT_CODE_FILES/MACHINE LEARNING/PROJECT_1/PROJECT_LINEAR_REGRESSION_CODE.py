# -*- coding: utf-8 -*-
'''Author -> Sahil'''
#Importing the Libraries

import pandas as pd
import matplotlib.pyplot as plt

#Reading the data from files
data = pd.read_csv("advertising.csv")
data.head()


#Visualizing the data sets
fig, axs = plt.subplots(1, 3, sharey = True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16,8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

#Creating X and Y for Linear Regression
feature_cols = ['TV']
X = data[feature_cols]
Y= data.Sales

#Importing Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

print(lr.intercept_)
print(lr.coef_)

result = (0.05546477*50) + 6.97
print(result)

#Creating a DataFrame with MIN and MAX Values of the Table
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()


preds = lr.predict(X_new)
preds


data.plot(kind = 'scatter', x='TV', y='Sales')

plt.plot(X_new,preds, c='red', linewidth = 3)


import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~TV', data = data).fit()
lm.conf_int()




#Finding the Probability Values
lm.pvalues


#Finding the R-Squared Values
lm.rsquared




#Multi Linear Regression
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
Y= data.Sales


lr = LinearRegression()
lr.fit(X, Y)


print(lr.intercept_)
print(lr.coef_)



lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper', data=data).fit()
lm.conf_int()
lm.summary()

























