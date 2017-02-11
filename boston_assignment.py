
# coding: utf-8

# ## Boston Housing Assignment
# 
# In this assignment you'll be using linear regression to estimate the cost of house in boston, using a well known dataset.
# 
# Goals:
# +  Measure the performance of the model I created using $R^{2}$ and MSE
# > Learn how to use sklearn.metrics.r2_score and sklearn.metrics.mean_squared_error
# +  Implement a new model using L2 regularization
# > Use sklearn.linear_model.Ridge or sklearn.linear_model.Lasso 
# +  Get the best model you can by optimizing the regularization parameter.   

# In[1]:

from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[2]:

bean = datasets.load_boston()
print(bean.DESCR)


# In[3]:

def load_boston():
    scaler = StandardScaler()
    boston = datasets.load_boston()
    X=boston.data
    y=boston.target
    X = scaler.fit_transform(X)
    return train_test_split(X,y)
    


# In[4]:

X_train, X_test, y_train, y_test = load_boston()


# In[5]:

X_train.shape


# In[6]:


clf = LinearRegression()
clf.fit(X_train, y_train)


# ### Making a Prediction
# X_test is our holdout set of data.  We know the answer (y_test) but the computer does not.   
# 
# Using the command below, I create a tuple for each observation, where I'm combining the real value (y_test) with
# the value our regressor predicts (clf.predict(X_test))
# 
# Use a similiar format to get your r2 and mse metrics working.  Using the [scikit learn api](http://scikit-learn.org/stable/modules/model_evaluation.html) if you need help!

# In[7]:

m1_estimated_results = list(zip(y_train, clf.predict(X_train)))
m1_test_results = list(zip (y_test, clf.predict(X_test)))


# In[8]:

from math import sqrt
## extract values from each sublist http://stackoverflow.com/questions/25050311/extract-first-item-of-each-sublist-in-python
m1MSE_train = mean_squared_error([item[0] for item in m1_estimated_results],[item[1] for item in m1_estimated_results])
#scaled RMSE for training set
m1RMSE_train = sqrt(m1MSE_train)


# In[9]:

#mean_squared_error(y_test, clf.predict(X_test))
m1MSE_test = mean_squared_error([item[0] for item in m1_test_results],[item[1] for item in m1_test_results])
#scaled RMSE for test set
m1RMSE_test = sqrt(m1MSE_test)


# In[10]:

#r2 for training set
m1R2_train = r2_score([item[0] for item in m1_estimated_results],[item[1] for item in m1_estimated_results])


# In[11]:

#r2 for test set
m1R2_test = r2_score([item[0] for item in m1_test_results],[item[1] for item in m1_test_results])


# In[12]:

#Model 1 results
## manipulate pandas dataframe http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html
results = pd.DataFrame({'R^2': [m1R2_train, m1R2_test],
                        'RMSE':[m1RMSE_train, m1RMSE_test]})
results = results.rename({0:'Model1_Training_Set', 1:'Model1_Test_Set'})
print("-----------------------------")
print("Original Linear Regression")
print(results)
results


# ### Initial Impressions of Model
# <li> R^2 looks pretty good with the model being able to explain 60-80 % of the variability of Boston House Prices
# <li> RMSE looks appropriate just glancing at scaled results eyeballing a 10+/-5% error in predictions
# <li> Good model fit and consistent between train and test sets initially indicate has low bias and low variance
# > inquire about R2 and RSME on test vs train sets (which one to use?)
# > inquire about how to unscale predicted values

# ### Ridge Model Analysis

# In[13]:

## Ridge & Lasso http://scikit-learn.org/stable/modules/linear_model.html
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
###opt_RSME = list()
###opt_R2 = list()
###for i in range(1000):
###    ridge = Ridge(alpha = (i/10))
###    ridge.fit(X_train, y_train)
###    ridge_estimated_results = list(zip(y_train, ridge.predict(X_train)))
###    ridge_test_results = list(zip(y_test, ridge.predict(X_test)))
###    ridgeMSE_test = mean_squared_error([item[0] for item in ridge_test_results],
###                                       [item[1] for item in ridge_test_results])
###    ridgeRMSE_test = sqrt(ridgeMSE_test)
###    ridgeR2_train = r2_score([item[0] for item in ridge_estimated_results],
###                             [item[1] for item in ridge_estimated_results])
###    opt_RSME.append(ridgeRMSE_test)
###    opt_R2.append(ridgeR2_train)
    
#Index of minimum RSME
###opt_RSME.index(min(opt_RSME))
#Index of maximum R^2
###opt_R2.index(max(opt_R2))


# In[14]:

#Test Ridge Model with alpha between 0 and 100 in 0.1 increments
ridge = RidgeCV(alphas = np.arange(0.0, 10.1, 0.1))
ridge.fit(X_train, y_train)
opt_alpha = ridge.alpha_
#Run Ridge Model with optimized Alpha
ridge = Ridge(alpha = opt_alpha)
ridge.fit(X_train, y_train)
ridge_estimated_results = list(zip(y_train, ridge.predict(X_train)))
ridge_test_results = list(zip(y_test, ridge.predict(X_test)))
ridgeMSE_train = mean_squared_error([item[0] for item in ridge_estimated_results],
                                   [item[1] for item in ridge_estimated_results])
ridgeRMSE_train = sqrt(ridgeMSE_train)
ridgeMSE_test = mean_squared_error([item[0] for item in ridge_test_results],
                                   [item[1] for item in ridge_test_results])
ridgeRMSE_test = sqrt(ridgeMSE_test)
ridgeR2_train = r2_score([item[0] for item in ridge_estimated_results],
                         [item[1] for item in ridge_estimated_results])
ridgeR2_test = r2_score([item[0] for item in ridge_test_results],
                         [item[1] for item in ridge_test_results])
ridge_results = pd.DataFrame({'R^2': [ridgeR2_train, ridgeR2_test],
                        'RMSE':[ridgeRMSE_train, ridgeRMSE_test]})
ridge_results = ridge_results.rename({0:'Ridge_Training_Set', 1:'Ridge_Test_Set'})
print("-----------------------------")
print("Ridge Regression with optimized alpha = ", opt_alpha)
print(ridge_results)
ridge_results


# ### Initial Impressions of Ridge Model
# <li> alpha = 0 produces highest R^2 and lowest RMSE for both training and test
# <li> produces same result of original linear regression
