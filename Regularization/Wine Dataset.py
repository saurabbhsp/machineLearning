
# coding: utf-8

# In[430]:


get_ipython().run_cell_magic('javascript', '', '<!-- Ignore this block -->\nIPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[431]:


#%config InlineBackend.figure_format = 'retina'

from __future__ import division
import pandas as pd
from itertools import product
import numpy as np
from math import sqrt, isnan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ## generatePearsonCoefficient Procedure
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f76ccfa7c2ed7f5b085115086107bbe25d329cec">
# For sample:-
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/bd1ccc2979b0fd1c1aec96e386f686ae874f9ec0">
# For selecting some features and for dropping others I am using Pearson's Coefficient. The value of Pearson's coefficient lies between [-1, 1] and tells how two features are related<br>
# <table>
# <tr><td>Strength of Association</td><td>Positive</td><td>Negative</td></tr><tr><td>Small</td><td>.1 to .3  </td><td>-0.1 to -0.3  </td></tr><tr><td>Medium</td><td>.3 to .5  </td><td>-0.3 to -0.5  </td></tr><tr><td>Large</td><td>.5 to 1.0 </td><td>-0.5 to -1.0  </td></tr></table>
# 

# In[432]:


"""Generate pearson's coefficient"""

def generatePearsonCoefficient(A, B):
    A = A - A.mean()
    B = B - B.mean()
    return ((A * B).sum())/(sqrt((A * A).sum()) * sqrt((B * B).sum())) 


# ## Feature scaling
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b0aa2e7d203db1526c577192f2d9102b718eafd5">

# In[433]:


def scaleFeature(x):
    mean = np.mean(x)
    stdDeviation = np.std(x)
    return x.apply(lambda y: ((y * 1.0) - mean)/(stdDeviation))


# ## SplitDataSet Procedure
# This method splits the dataset into trainset and testset based upon the trainSetSize value. For splitting the dataset, I am using pandas.sample to split the data. This gives me trainset. For testset I am calculating complement of the trainset. This I am doing by droping the index present in training set.

# In[434]:


"""Splits the provided pandas dataframe into training and test dataset"""
def splitDataSet(inputDataframe, trainSetSize):
    
        trainSet = inputDataframe.sample(frac = trainSetSize)
        testSet = inputDataframe.drop(trainSet.index)
        return trainSet, testSet


# ## RMSE procedure
# Will calculate root mean squared error for given Ytrue values and YPrediction.
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/fc187c3557d633423444d4c80a4a50cd6ecc3dd4">
# 

# In[435]:


"""Model accuracy estimator RMSE"""

def RMSE(yTrue, yPrediction):
    n = yTrue.shape[0]
    return sqrt((1.0) * np.sum(np.square((yTrue - yPrediction))))/n


# ## Regularization
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/d55221bf8c9b730ff7c4eddddb9473af47bb1d1c">

# ### L2 loss
# L2 loss or Tikhonov regularization
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/7328255ad4abce052b3f6f39c43e974808d0cdb6">
# Caution: Do not regularize B0 or bias term

# In[436]:


def l2Loss(regularizationParameter, weight):
    loss = 2 * regularizationParameter * weight
    "Remove impact on bias"
    loss[0] = 0
    return loss


# ## miniBatchGradientDescent

# In[437]:


"""If no step length controller is provided then values of alpha will be taken as step length.
Else the step length controller will be used. Additional parameters to the controller are
provided by stepLengthControllerParameters"""

def miniBatchGradientDescent(xTrain, yTrain, xTest, yTest, beta, epochs = 5, 
                             batchSize = 50, verbose = 0, alpha = 1.1e-5, 
                             regularizationParameter = 9e-2, regularization = None):
    
    xTrain = np.insert(xTrain, 0, 1, axis = 1)
    xTest = np.insert(xTest, 0, 1, axis = 1)
    
    xTrain = xTrain * 1.0
    yTrain = yTrain * 1.0

    xTest = xTest * 1.0
    yTest = yTest * 1.0

    
    """For plotting graphs"""
    trainRMSE = []
    testRMSE = []
    
   # if stepLengthController != None:
   #     print("Warning using stepLengthController alpha values will be rewritten")
    
    indices = np.array(range(0, len(xTrain)))
    for i in range(0, epochs):
        
        if verbose:
            print("Epoch-"+str(i))
        
        
        """Shuffle the indices"""                        
        np.random.shuffle(indices)
        
        """Will split. May be uneven"""
        batches = np.array_split(indices, batchSize)
        
        if verbose:
            print("Total batches created"+str(len(batches)))
        
        index = 0
        while index < len(xTrain):
            
            batch = indices[index : index + batchSize]
            index = index + batchSize
            
            """Select required x and y subsets"""
            x = np.take(xTrain, batch, axis = 0)
            y = np.take(yTrain, batch, axis = 0)
            
            prediction = np.dot(beta, x.T)
            residual = prediction - y
            gradient = np.dot(residual, x) * 2
        
            
            """Gradient descent"""
            regFactor = 0
            if regularization != None:
                regFactor = regularization(regularizationParameter, beta)
                
            beta = beta - (alpha * (gradient + regFactor))
                
        
        if verbose:
            print beta
        
        """Calculating RMSE for train and test set"""
        xTrainPrediction = np.dot(beta, xTrain.T)
        xTestPrediction = np.dot(beta, xTest.T)
        
        trainRMSE.append(RMSE(yTrain, xTrainPrediction))
        testRMSE.append(RMSE(yTest, xTestPrediction))
    return beta, trainRMSE, testRMSE


# ## kFoldAnalysis

# In[438]:


def kFoldAnalysis(xTrain, yTrain, model, modelParameters, nFolds):
    
    indices = np.array(range(0, len(xTrain)))
    folds = np.array_split(indices, nFolds)
    
    analysisMetricList = []
    trainRMSEList = []
    testRMSEList = []
    
    for i in range(0, len(folds)):
        validationSet = folds[i]
        
        """Set difference"""
        trainSet = np.setdiff1d(indices, validationSet)
        
        modelParameters['xTrain'] = np.take(xTrain, trainSet, axis = 0)
        modelParameters['yTrain'] = np.take(yTrain, trainSet, axis = 0)
        modelParameters['xTest'] = np.take(xTrain, validationSet, axis = 0)
        modelParameters['yTest'] = np.take(yTrain, validationSet, axis = 0)
        
        modelParams, trainRMSE, testRMSE = model(**modelParameters)
        analysisMetricList.append(testRMSE[-1])
        trainRMSEList.append(trainRMSE)
        testRMSEList.append(testRMSE)
        
    return modelParams, trainRMSEList, testRMSEList, analysisMetricList
    


# ## GridSearch

# In[439]:


def gridSearch(xTrain, yTrain, xTest, yTest, model, modelParameters, hyperParameters, 
               nFolds = 1, reTrain = True, plotGraphs = False):
    
    """For storing is the best parameters"""
    leastRMSE = None
    bestModel = None
    bestHyperParams = None
    
    """Generate the parameter grid"""
    parameterGrid = []
    gridKeys = []
    
    parameterGrid = list(product(*hyperParameters.values()))
    hyperParameterKeys = hyperParameters.keys()
    
    """For plottong graphs"""
    if plotGraphs:
        plt.close()
        plotHeight = 10
        plotWidth = 20
        index = 0
        fig, axs = plt.subplots(len(parameterGrid), 2, figsize=(plotWidth, plotHeight * len(parameterGrid)))
        
          
        fig = plt.figure()
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax = fig.add_subplot(111, projection='3d')
            
        
    """Grid search for cartesian product of hyperParameters"""    
    for parameterMesh in parameterGrid:
        hyperParameterMesh = {}
        for k,v in zip(hyperParameterKeys, parameterMesh):
            hyperParameterMesh[k] = v
        
        """Combine model Parameters"""
        updatedParam = modelParameters.copy()
        updatedParam.update(hyperParameterMesh)
        
        """Perform grid search with cross validation"""
        if nFolds > 1:
            modelParams, trainRMSEList, testRMSEList, analysisMetricList = kFoldAnalysis(model = model,
                                                                                        xTrain = xTrain,
                                                                                        yTrain = yTrain,
                                                                                        nFolds = nFolds,
                                                                                        modelParameters = updatedParam)  
            
            
            """For storing best model"""
            avg = np.average(analysisMetricList)
            if leastRMSE == None or avg < leastRMSE:
                leastRMSE = avg
                bestModel = modelParams
                bestHyperParams = hyperParameterMesh
            
            """For plotting"""
            if plotGraphs:
                foldIndex = 1
                
                ax.scatter(hyperParameterMesh['alpha'], hyperParameterMesh['regularizationParameter'], 
                           avg, marker = 'o', label = str(hyperParameterMesh))
                              
                for train, test in zip(trainRMSEList, testRMSEList):
                    axs[index][0].plot(train, label = "Fold-" + str(foldIndex))
                    axs[index][1].plot(test, label = "Fold-" + str(foldIndex))
                    foldIndex = foldIndex + 1
                
                axs[index][0].legend()
                axs[index][0].grid()
                
                axs[index][1].legend()
                axs[index][1].grid()
                
                axs[index][0].set_title("Train set for " + str(hyperParameterMesh))
                axs[index][1].set_title("Validation set for " + str(hyperParameterMesh))
                    
                
                index = index + 1
            
                
            """Perform only grid search and no cross validation. Test set will be used for validation"""    
        else:
            trainedModel, trainRMSE, testRMSE = model(xTrain, yTrain, xTest, yTest, **updatedParam)
            
            """For storing best model"""
            if leastRMSE == None or testRMSE[-1] < leastRMSE:
                leastRMSE = testRMSE[-1]
                bestModel = trainedModel
                bestHyperParams = hyperParameterMesh
            
            """For plotting graphs"""
            if plotGraphs:
                axs[index][0].plot(trainRMSE, label = "Training set RMSE for " + str(hyperParameterMesh))
                axs[index][0].legend()
                axs[index][0].grid()
                axs[index][1].plot(testRMSE, label = "Test set RMSE for " + str(hyperParameterMesh))
                axs[index][1].legend()
                axs[index][1].grid()
                index = index + 1
    
    if plotGraphs:
        ax.legend()
        ax.set_xlabel('alpha')
        ax.set_ylabel('regularizationParameter')
        ax.set_zlabel('RMSE')
        plt.show()
        plt.close()
    
    if reTrain:
        
        """Combine model Parameters"""
        updatedParam = modelParameters.copy()
        updatedParam.update(bestHyperParams)

        bestModel, trainRMSE, testRMSE = model(xTrain, yTrain, xTest, yTest, **updatedParam)
        
        if plotGraphs:
            plt.close()
            plotHeight = 10
            plotWidth = 20
            fig, axs = plt.subplots(1, 2, figsize = (plotWidth, plotHeight)) 
            
            plt.suptitle("Best model")

            axs[0].plot(trainRMSE, label = "Training set RMSE for " + str(bestHyperParams))
            axs[0].legend()
            axs[0].grid()
            axs[1].plot(testRMSE, label = "Test set RMSE for " + str(bestHyperParams))
            axs[1].legend()
            axs[1].grid()
        
            plt.show()
        
        
      
        
        
    
    return bestModel, bestHyperParams


# ## Get quadratic variables

# In[440]:


def getQuadraticDerivedVariables(inputDataFrame, keys, degree):
    newKeys = []
    for key in keys:
        inputDataFrame[key+" Degree-" + str(degree)] = inputDataFrame[key] ** degree
        newKeys.append(key+" Degree-" + str(degree))
    return newKeys


# ## Make Prediction

# In[441]:


def predict(modelParameters, x):
    x = np.insert(x, 0, 1, axis = 1)
    x = x * 1
    return np.dot(modelParameters, x.T)


# ## Wine dataset

# ### Load dataset

# In[442]:


""" File path change accordingly"""
directoryPath = "Data"

wineData = pd.read_csv(directoryPath+"/winequality-red.csv", sep=";")
wineData.head()


# In[443]:


wineData.describe().T


# In[444]:


wineData.dropna(inplace = True)
wineData.head()


# In[445]:


totalFeatureSet = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                   'sulphates', 'alcohol']


# ### Using quadratic model

# In[446]:


derivedFeatures = getQuadraticDerivedVariables(wineData, totalFeatureSet, 2)
totalFeatureSet.extend(derivedFeatures)
wineData.head()


# ### Feature scaling

# In[447]:


for feature in wineData:
    if feature != "quality":
        wineData[feature] = scaleFeature(wineData[feature])

wineData.head()


# ### Checking corelation

# In[448]:


for column in wineData:
    if column != "quality":
        print("The corelation between " + column +" vs quality is " + 
              str(generatePearsonCoefficient(wineData[column], wineData['quality'])))


# We will drop features with low corelation

# In[449]:


totalFeatureSet.remove('residual sugar')
totalFeatureSet.remove('free sulfur dioxide')
totalFeatureSet.remove('pH')
totalFeatureSet.remove('residual sugar Degree-2')
totalFeatureSet.remove('chlorides Degree-2')
totalFeatureSet.remove('free sulfur dioxide Degree-2')
totalFeatureSet.remove('pH Degree-2')
print totalFeatureSet


# In[450]:


wineData.head()


# ### Split dataset

# In[451]:


trainSet, testSet = splitDataSet(wineData, 0.8)


# In[452]:


xTrain = trainSet.as_matrix(columns = totalFeatureSet)
xTest = testSet.as_matrix(columns = totalFeatureSet)

yTrain = trainSet['quality']
yTest = testSet['quality']


# ### Regularized vs non regularized model

# In[453]:


plt.close()


modelParameters = {"beta":np.zeros(xTrain.shape[1] + 1), "epochs":100, 
                   "batchSize":50, "verbose":0}

alphaList = [4.0e-3, 5.9e-4, 9.1e-5]
regularizationParameterList = [9.9e-9, 1.9e-7, 5.9e-3]

"""For plotting traph"""
plotHeight = 10
plotWidth = 20
index = 0
fig, axs = plt.subplots(len(alphaList), 2, figsize=(plotWidth, plotHeight * len(alphaList)))

for alpha, regularizationParameter in zip(alphaList, regularizationParameterList):
    betaNonReg, trainRMSENonReg, testRMSENonReg = miniBatchGradientDescent(xTrain, yTrain, xTest, yTest,
                                                                           alpha = alpha, **modelParameters)
    
    betaReg, trainRMSEReg, testRMSEReg = miniBatchGradientDescent(xTrain, yTrain, xTest, yTest,
                                                                           alpha = alpha, regularization = l2Loss,
                                                                      regularizationParameter = regularizationParameter,
                                                                      **modelParameters)
    axs[index][0].set_xlabel("Iterations")
    axs[index][1].set_xlabel("Iterations")

    axs[index][0].set_ylabel("RMSE")
    axs[index][1].set_ylabel("RMSE")
    
    
    axs[index][0].plot(trainRMSENonReg, label = "Non Regularized")
    axs[index][0].plot(trainRMSEReg, label = "Regularized")
    axs[index][0].grid()
    axs[index][0].legend()

    
    axs[index][1].plot(testRMSENonReg, label = "Non Regularized")
    axs[index][1].plot(testRMSEReg, label = "Regularized")
    axs[index][1].grid()
    axs[index][1].legend()
    index = index + 1

plt.show()    
    


# A bad value of regularization parameter results into diverging rather than converging graph. This is evident in the third graph.
# It is very difficult to calculate hyper parameters in this way.
# Hence we use grid search.

# ### Using grid search with k-fold 
# Here plotting takes some time. Make plotGraphs as false to avoid plotting of graphs

# In[454]:


"""Parameter grid for searching"""

gridParameters = {
    "alpha":[5.0e-5, 5.5e-5, 5.9e-5, 5.0e-6, 5.5e-7, 5.9e-8],
    "regularizationParameter":[9e-3, 9e-4, 9e-5, 9e-6]
}


modelParameters = {
    "beta":np.zeros(xTrain.shape[1] + 1), 
    "epochs":100, 
    "batchSize":50, 
    "verbose":0,
    "regularization" : l2Loss
}

model, hyperParams = gridSearch(xTrain, yTrain, xTest, yTest, 
           miniBatchGradientDescent, modelParameters, 
           gridParameters, nFolds = 5, reTrain = True, plotGraphs = True)


# Here grid search is used to calculate best model. All possible combinations of hyperparameters are calculated. Then kfold analyis is done. The one with least error is selected as best model
