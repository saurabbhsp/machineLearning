
# coding: utf-8

# In[101]:


get_ipython().run_cell_magic('javascript', '', '<!-- Ignore this block -->\nIPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[102]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[103]:


""" File path change accordingly"""
directoryPath = "Data"

wineData = pd.read_csv(directoryPath+"/winequality-red.csv", sep=";")
wineData.head()


# In[104]:


"""Dropping free sulfur dioxide"""
selectedFeatures = list(wineData)
selectedFeatures.remove('quality')
selectedFeatures.remove('free sulfur dioxide')


# ## Split data

# In[105]:


from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(wineData[selectedFeatures], wineData['quality'],
                                                test_size = 0.2)

print("Train set size " +str(len(xTrain)))
print("Test set size " +str(len(xTest)))


# # Nearest Neighbour

# ## Create a pipeline for classification

# In[106]:


classificationPipeline = Pipeline([
    ("featureScaling", StandardScaler()),
    ("classification", KNeighborsClassifier())
])


# ## Define grid for grid search

# In[107]:


parameterGrid = {
    "classification__n_neighbors":range(5,25),
    "classification__weights":["uniform", "distance"],
    "classification__algorithm":["brute"]
    
}


# ## Gridsearch

# In[108]:


from sklearn.model_selection import GridSearchCV


modelKNN = GridSearchCV(classificationPipeline, cv = 3, n_jobs = 3, param_grid = parameterGrid,
             verbose = 1, refit = True)

modelKNN.fit(xTrain * 1.0, yTrain * 1.0)


# In[109]:


print("Best model " + str(modelKNN.grid_scores_))


# In[110]:


plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(modelKNN.cv_results_["mean_test_score"])
plt.plot(modelKNN.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()


# # KDTree
# It is a space partioning algorithm for KNN. Optimizes the algorithm by reducing the search area for algorithm
# <br>
#     function kdtree (list of points pointList, int depth)<br>
# {<br>
# 
#     // Select axis based on depth so that axis cycles through all valid values
#     var int axis := depth mod k;
#         
#     // Sort point list and choose median as pivot element
#     select median by axis from pointList;
#         
#     // Create node and construct subtree
#     node.location := median;
#     node.leftChild := kdtree(points in pointList before median, depth+1);
#     node.rightChild := kdtree(points in pointList after median, depth+1);
#     return node;
# }

# In[111]:


parameterGrid = {
    "classification__n_neighbors":range(5,25),
    "classification__weights":["uniform", "distance"],
    "classification__algorithm":["kd_tree"],
    "classification__leaf_size":[31]
}


# ## Grid search

# In[112]:


modelKNNKDTree = GridSearchCV(classificationPipeline, cv = 3, n_jobs = 3, param_grid = parameterGrid,
             verbose = 1, refit = True)

modelKNNKDTree.fit(xTrain * 1.0, yTrain * 1.0)


# In[113]:


print("Best model " + str(modelKNNKDTree.grid_scores_))


# In[114]:


plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(modelKNNKDTree.cv_results_["mean_test_score"])
plt.plot(modelKNNKDTree.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()


# # Ball Tree
# It is a space partioning algorithm for KNN. Optimizes the algorithm by reducing the search area for algorithm. It generates hypersphere.

# In[115]:


parameterGrid = {
    "classification__n_neighbors":range(5,25),
    "classification__weights":["uniform", "distance"],
    "classification__algorithm":["ball_tree"],
    "classification__leaf_size":[31]
}


# In[116]:


modelKNNBallTree = GridSearchCV(classificationPipeline, cv = 3, n_jobs = 3, param_grid = parameterGrid,
             verbose = 1, refit = True)

modelKNNBallTree.fit(xTrain * 1.0, yTrain * 1.0)


# In[117]:


print("Best model " + str(modelKNNBallTree.grid_scores_))


# In[118]:


plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(modelKNNBallTree.cv_results_["mean_test_score"])
plt.plot(modelKNNBallTree.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()


# ## Generating classification report for all best models

# In[119]:


from sklearn.metrics import classification_report
print("KNN")
print(classification_report(yTest, modelKNN.predict(xTest)))

print("KNN KD Tree")
print(classification_report(yTest, modelKNNKDTree.predict(xTest)))

print("KNN Ball Tree")
print(classification_report(yTest, modelKNNBallTree.predict(xTest)))


# In[120]:


"""Checking best score"""
print(modelKNN.best_score_)
print(modelKNNKDTree.best_score_)
print(modelKNNBallTree.best_score_)


# In[121]:


get_ipython().run_line_magic('timeit', 'modelKNN.predict(xTest)')


# In[122]:


get_ipython().run_line_magic('timeit', 'modelKNNKDTree.predict(xTest)')


# In[123]:


get_ipython().run_line_magic('timeit', 'modelKNNBallTree.predict(xTest)')


# In[124]:


irisData = pd.read_csv(directoryPath+"/iris.data", names = ["sepalLength", 
                                                            "sepalWidth", "petalLength", "petalWidth", "target"])
irisData.head()


# In[125]:


from sklearn import preprocessing


encoder = preprocessing.LabelEncoder()
irisData['target'] = encoder.fit_transform(irisData['target'])


# In[126]:


xTrain, xTest, yTrain, yTest = train_test_split(irisData[["sepalLength","sepalWidth", 
                                                          "petalLength", "petalWidth"]], irisData['target'],
                                                test_size = 0.2)

print("Train set size " +str(len(xTrain)))
print("Test set size " +str(len(xTest)))


# In[127]:


classificationPipeline = Pipeline([
    ("featureScaling", StandardScaler()),
    ("classification", KNeighborsClassifier())
])

parameterGrid = {
    "classification__n_neighbors":range(5,25),
    "classification__weights":["uniform", "distance"],
    "classification__algorithm":["brute"]
    
}


modelKNN = GridSearchCV(classificationPipeline, cv = 3, n_jobs = 3, param_grid = parameterGrid,
             verbose = 1, refit = True)

modelKNN.fit(xTrain * 1.0, yTrain * 1.0)


# In[128]:


print("Best model " + str(modelKNN.grid_scores_))


# In[129]:


plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(modelKNN.cv_results_["mean_test_score"])
plt.plot(modelKNN.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()


# In[130]:


parameterGrid = {
    "classification__n_neighbors":range(5,25),
    "classification__weights":["uniform", "distance"],
    "classification__algorithm":["kd_tree"],
    "classification__leaf_size":[31]
}


# In[131]:


modelKNNKDTree = GridSearchCV(classificationPipeline, cv = 3, n_jobs = 3, param_grid = parameterGrid,
             verbose = 1, refit = True)

modelKNNKDTree.fit(xTrain * 1.0, yTrain * 1.0)


# In[132]:


plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(modelKNNKDTree.cv_results_["mean_test_score"])
plt.plot(modelKNNKDTree.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()


# In[133]:


parameterGrid = {
    "classification__n_neighbors":range(5,25),
    "classification__weights":["uniform", "distance"],
    "classification__algorithm":["ball_tree"],
    "classification__leaf_size":[31]
}


# In[134]:


modelKNNBallTree = GridSearchCV(classificationPipeline, cv = 3, n_jobs = 3, param_grid = parameterGrid,
             verbose = 1, refit = True)

modelKNNBallTree.fit(xTrain * 1.0, yTrain * 1.0)


# In[135]:


print("Best model " + str(modelKNNBallTree.grid_scores_))


# In[136]:


plt.figure(figsize = (30, 10))
plt.title("Mean score")
plt.plot(modelKNNBallTree.cv_results_["mean_test_score"])
plt.plot(modelKNNBallTree.cv_results_["mean_test_score"], "bo", label = "model with different hyper parameters")
plt.xlabel('Models', fontsize = 15)
plt.ylabel('Error', fontsize = 15)
plt.legend()
plt.show()


# In[137]:


print("KNN")
print(classification_report(yTest, modelKNN.predict(xTest)))

print("KNN KD Tree")
print(classification_report(yTest, modelKNNKDTree.predict(xTest)))

print("KNN Ball Tree")
print(classification_report(yTest, modelKNNBallTree.predict(xTest)))


# In[138]:


"""Checking best score"""
print(modelKNN.best_score_)
print(modelKNNKDTree.best_score_)
print(modelKNNBallTree.best_score_)


# Here we can say that for both the data sets the best score remains same for brute force, KDTree and BallTree. There is same precison and recall for models. This is because KDTree and BallTree are space partitioning algorithms. They reduce the no of comparasions required for K-NN algorithm. Yet the core algorithm is same, hence the results are same. Both methods speed up discovery of neighbours

# In[143]:


get_ipython().run_line_magic('timeit', 'modelKNN.predict(xTest)')


# In[142]:


get_ipython().run_line_magic('timeit', 'modelKNNKDTree.predict(xTest)')


# In[141]:


get_ipython().run_line_magic('timeit', 'modelKNNBallTree.predict(xTest)')

