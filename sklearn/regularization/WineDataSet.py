
# coding: utf-8

# In[51]:


get_ipython().run_cell_magic('javascript', '', '<!-- Ignore this block -->\nIPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[52]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from __future__ import division
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Load Data

# In[53]:


""" File path change accordingly"""
directoryPath = "Data"

wineData = pd.read_csv(directoryPath+"/winequality-red.csv", sep=";")
wineData.head()


# In[54]:


wineData.describe().T


# ## Visualize data

# In[55]:


wineData.hist(figsize = (20, 20))
plt.show()


# In[56]:


wineData.plot(kind='density', subplots=True, layout=(4,3), sharex=False, figsize = (20, 20))
plt.show()


# In[57]:


correlation = wineData.corr(method = "spearman")
correlation


# In[58]:


labels = list(wineData)
fig, ax = plt.subplots(1, 1, figsize=(25, 25))
ax.matshow(correlation)

ticks = range(0, len(labels))
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()


# There is a very high corelation between total sulfur dioxide and free sulfur dioxide. One of these features can be dropped. Correlation of free sulfur dioxide is very less with quality as compared to total sulfur dioxide. So we will drop free sulfur dioxide. 

# In[59]:


selectedFeatures = list(wineData)
selectedFeatures.remove('quality')
selectedFeatures.remove('free sulfur dioxide')


# ## Split dataset

# In[60]:


from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(wineData[selectedFeatures], wineData['quality'],
                                                test_size=0.2)

print("Train set size " +str(len(xTrain)))
print("Test set size " +str(len(xTest)))


# ## Creating a pipeline
# Here the pipeline included two steps.<br>
# 1) Feature scaling<br>
# 2) Training classification algorithm<br>

# In[61]:


from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""Pipeline will take care of transformation and fitting the model on it
Here the first step is to perform feature scaling
Second step is to fit the model for stochastic gradient descent Regression"""

trainingPipeline = Pipeline([
    ("featureScaling", StandardScaler()),
    ("regression", SGDRegressor())
])


"""No penalty and constant learning rate"""
pipelineParameters = {"regression__learning_rate":"constant", "regression__penalty":None}


# ## Training the model without regularization

# In[62]:


from sklearn.metrics import mean_squared_error
"""Training without regularization"""


learningRates = np.arange(0.1, 0.9, 0.05)
trainRMSENoReg = []
testRMSENoReg = []

for learningRate in learningRates:
    pipelineParameters["regression__eta0"] = learningRate
    trainingPipeline.set_params(** pipelineParameters)
    trainingPipeline.fit(xTrain, yTrain)
    
    trainRMSENoReg.append(sqrt(mean_squared_error(yTrain, trainingPipeline.predict(xTrain))))
    testRMSENoReg.append(sqrt(mean_squared_error(yTest, trainingPipeline.predict(xTest))))


# In[63]:


"""Plot the trained model"""

fig, axs = plt.subplots(1, 2, figsize=(25, 10))

fig.suptitle("Alpha vs RMSE, No regularization", fontsize = 20)
axs[0].plot(learningRates, trainRMSENoReg)
axs[0].plot(learningRates, trainRMSENoReg, "ro")
axs[0].grid()
axs[0].set_xticks(learningRates)
axs[0].set_xlabel('Alpha', fontsize = 15)
axs[0].set_ylabel('RMSE', fontsize = 15)
axs[0].set_title("Training set", fontsize = 15)

axs[1].plot(learningRates, testRMSENoReg)
axs[1].plot(learningRates, testRMSENoReg, "ro")
axs[1].grid()
axs[1].set_xticks(learningRates)
axs[1].set_xlabel('Alpha', fontsize = 15)
axs[1].set_ylabel('RMSE', fontsize = 15)
axs[1].set_title("Test set", fontsize = 15)
                  
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(25, 10))
axs.plot(trainRMSENoReg, label = "Train set RMSE")
axs.plot(testRMSENoReg, label = "Test set RMSE")
axs.grid()
axs.legend()
plt.show()


# Here we are not using any regularization. We can see that the error on train set is less than test error.

# ## Training the model with L2 loss (Ridge Regression)

# In[64]:


pipelineParameters = {"regression__learning_rate":"constant", "regression__penalty":"l2",
                     "regression__alpha":0.01}


# In[65]:


"""Training with L2 regularization"""


trainRMSERegL2 = []
testRMSERegL2 = []

for learningRate in learningRates:
    pipelineParameters["regression__eta0"] = learningRate
    trainingPipeline.set_params(** pipelineParameters)
    trainingPipeline.fit(xTrain, yTrain)
    
    trainRMSERegL2.append(sqrt(mean_squared_error(yTrain, trainingPipeline.predict(xTrain))))
    testRMSERegL2.append(sqrt(mean_squared_error(yTest, trainingPipeline.predict(xTest))))


# In[66]:


fig, axs = plt.subplots(1, 2, figsize=(25, 10))

fig.suptitle("Alpha vs RMSE, L2 penalty, alpha = 1e-2", fontsize = 20)
axs[0].plot(learningRates, trainRMSERegL2)
axs[0].plot(learningRates, trainRMSERegL2, "ro")
axs[0].grid()
axs[0].set_xticks(learningRates)
axs[0].set_xlabel('Alpha', fontsize = 15)
axs[0].set_ylabel('RMSE', fontsize = 15)
axs[0].set_title("Training set", fontsize = 15)

axs[1].plot(learningRates, testRMSERegL2)
axs[1].plot(learningRates, testRMSERegL2, "ro")
axs[1].grid()
axs[1].set_xticks(learningRates)
axs[1].set_xlabel('Alpha', fontsize = 15)
axs[1].set_ylabel('RMSE', fontsize = 15)
axs[1].set_title("Test set", fontsize = 15)
                  
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(25, 10))
axs.plot(trainRMSERegL2, label = "Train set RMSE")
axs.plot(testRMSERegL2, label = "Test set RMSE")
axs.legend()
axs.grid()
plt.show()


# The behaviour is similar to what we saw above.

# ## Training the model with L1 loss (Lasso Regression)

# In[67]:


pipelineParameters = {"regression__learning_rate":"constant", "regression__penalty":"l1",
                     "regression__alpha":0.001}


# In[68]:


"""Training with L1 regularization"""


trainRMSERegL1 = []
testRMSERegL1 = []

for learningRate in learningRates:
    pipelineParameters["regression__eta0"] = learningRate
    trainingPipeline.set_params(** pipelineParameters)
    trainingPipeline.fit(xTrain, yTrain)
    
    trainRMSERegL1.append(sqrt(mean_squared_error(yTrain, trainingPipeline.predict(xTrain))))
    testRMSERegL1.append(sqrt(mean_squared_error(yTest, trainingPipeline.predict(xTest))))


# In[69]:


fig, axs = plt.subplots(1, 2, figsize=(25, 10))

fig.suptitle("Alpha vs RMSE, L1 penalty, alpha = 1e-3", fontsize = 20)
axs[0].plot(learningRates, trainRMSERegL1)
axs[0].plot(learningRates, trainRMSERegL1, "ro")
axs[0].grid()
axs[0].set_xticks(learningRates)
axs[0].set_xlabel('Alpha', fontsize = 15)
axs[0].set_ylabel('RMSE', fontsize = 15)
axs[0].set_title("Training set", fontsize = 15)

axs[1].plot(learningRates, testRMSERegL1)
axs[1].plot(learningRates, testRMSERegL1, "ro")
axs[1].grid()
axs[1].set_xticks(learningRates)
axs[1].set_xlabel('Alpha', fontsize = 15)
axs[1].set_ylabel('RMSE', fontsize = 15)
axs[1].set_title("Test set", fontsize = 15)
                  
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(25, 10))
axs.plot(trainRMSERegL1, label = "Train set RMSE")
axs.plot(testRMSERegL1, label = "Test set RMSE")
axs.grid()
axs.legend()
plt.show()


# ## Comparing regularized vs Non Regularized

# In[70]:


fig, axs = plt.subplots(2, 1, figsize=(25, 20))

fig.suptitle("Alpha vs RMSE", fontsize = 20)
axs[0].plot(learningRates, trainRMSERegL1, label = "L1 Regularized")
axs[0].plot(learningRates, trainRMSERegL1, "o")

axs[0].plot(learningRates, trainRMSERegL2, label = "L2 Regularized")
axs[0].plot(learningRates, trainRMSERegL2, "o")

axs[0].plot(learningRates, trainRMSENoReg, label = "Non Regularized")
axs[0].plot(learningRates, trainRMSENoReg, "o")

axs[0].grid()
axs[0].set_xticks(learningRates)
axs[0].set_xlabel('Alpha', fontsize = 15)
axs[0].set_ylabel('RMSE', fontsize = 15)
axs[0].set_title("Training set", fontsize = 15)
axs[0].legend(prop={'size': 15})

axs[1].plot(learningRates, testRMSERegL1, label = "L1 Regularized")
axs[1].plot(learningRates, testRMSERegL1, "ro")

axs[1].plot(learningRates, testRMSERegL2, label = "L2 Regularized")
axs[1].plot(learningRates, testRMSERegL2, "ro")

axs[1].plot(learningRates, testRMSENoReg, label = "Non Regularized")
axs[1].plot(learningRates, testRMSENoReg, "ro")

axs[1].grid()
axs[1].set_xticks(learningRates)
axs[1].set_xlabel('Alpha', fontsize = 15)
axs[1].set_ylabel('RMSE', fontsize = 15)
axs[1].set_title("Test set", fontsize = 15)
axs[1].legend(prop={'size': 15})
                  
plt.show()


# Without adding the regularization term the non regularized model is performing better on train set when compared with L1(Lasso) and L2(Ridge) regularized model. This is because of the model tends to <b>overfit</b>. It performs good on training set. But the second graph shows the results on test set. We see that model performs poorly on test set when compared to L1 and L2 regularized.<br>
# <b>Underfitting</b> When the model is too simple to capture the trend in the data. This is called underfitting.

# ## Grid Search

# In[79]:


from sklearn.model_selection import GridSearchCV

"""Gridsearch is best way to find optimal hyper parameters"""

"""Grid for searching"""
param_grid = [
    {
        "regression__learning_rate" : ["constant"],
        "regression__eta0" :  np.arange(0.1, 0.9, 0.05),
        "regression__penalty" : [None] 
    },
    {
        "regression__learning_rate" : ["constant"],
        "regression__penalty" : ["l1"],
        "regression__alpha" : np.arange(0.1, 0.9, 0.05),
        "regression__eta0" :  np.arange(0.1, 0.9, 0.05)
    },
    {
        "regression__learning_rate" : ["constant"],
        "regression__penalty" : ["l2"],
        "regression__alpha" : np.arange(0.1, 0.9, 0.05),
        "regression__eta0" : np.arange(0.1, 0.9, 0.05)
    }
]


# In[72]:


gridNonRegularized = GridSearchCV(trainingPipeline, cv = 3, n_jobs = 3, param_grid = param_grid[0], verbose = 1, refit = True)
gridNonRegularized.fit(xTrain, yTrain)
print("Best model for given parameter grid \n" + str(gridNonRegularized.best_params_) + "\n")

plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(gridNonRegularized.cv_results_["mean_test_score"])
plt.plot(gridNonRegularized.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.plot(gridNonRegularized.best_score_, "ro")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()

print("Best score " + str(gridNonRegularized.best_score_))


# In[73]:


gridL1 = GridSearchCV(trainingPipeline, cv = 3, n_jobs = 3, param_grid = param_grid[2], verbose = 1, refit = True)
gridL1.fit(xTrain, yTrain)
print("Best model for given parameter grid \n" + str(gridL1.best_params_) + "\n")

plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(gridL1.cv_results_["mean_test_score"])
plt.plot(gridL1.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.plot(gridL1.best_score_, "ro")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()

print("Best score " + str(gridL1.best_score_))


# In[74]:


gridL2 = GridSearchCV(trainingPipeline, cv = 3, n_jobs = 3, param_grid = param_grid[1], verbose = 1, refit = True)
gridL2.fit(xTrain, yTrain)
print("Best model for given parameter grid \n" + str(gridL2.best_params_) + "\n")

plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(gridL2.cv_results_["mean_test_score"])
plt.plot(gridL2.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.plot(gridL2.best_score_, "ro")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()

print("Best score " + str(gridL2.best_score_))


# In[83]:


from sklearn.model_selection import cross_val_score

scoreNonReg = cross_val_score(trainingPipeline.set_params(**gridNonRegularized.best_params_), xTrain, yTrain)
scoreL1 = cross_val_score(trainingPipeline.set_params(**gridL1.best_params_), xTrain, yTrain)
scoreL2 = cross_val_score(trainingPipeline.set_params(**gridL2.best_params_), xTrain, yTrain)


plt.boxplot([scoreL2, scoreL1])
plt.title("L2 and L1 R^2 score boxplot")
plt.show()
plt.boxplot([scoreNonReg])
plt.title("Non regularized R^2 score boxplot")
plt.show()


# PS:- MSE scores are negative https://github.com/scikit-learn/scikit-learn/issues/2439
# <br>From boxplot we can see that the spread of error is very less for L1 and L2 regularized model. L1 is performing better among these two. The non regularized is not performing so good. The spread of error is also more.
