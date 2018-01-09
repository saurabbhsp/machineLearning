
# coding: utf-8

# # Model

# In[36]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import groupby
from math import sqrt


# ### Calculate mode of the data

# In[37]:


def getMode(x):
    frequency = groupby(Counter(x).most_common(), lambda x:x[1])
    mode = [val for val,count in frequency.next()[1]]
    return mode


# ## Distance measure

# In[38]:


class DistanceMeasure:
    
    @staticmethod
    def EuclidianDistance(a, b):
        return np.sqrt(np.sum((a - b)**2))
    
    @staticmethod
    def CosineDistance(a, b):
        return 1 - ( (1.0 * np.sum(a*b)) / ( np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)) ))
        


# ## InvSortedLinkedList

# In[39]:


"""Node has been designed for storing targetValue and distance"""
class Node:
    data = None
    payload = None
    nextNode = None
        
class InvSortedLinkedList:
    head = None
    tail = None
    
    def insert(self, node):
        """First insertion"""
        if self.head == None:
            self.head = node
            self.tail = node
        else:
            """Next insertions"""
            
            """Insertion at head"""
            if node.data > self.head.data:
                node.nextNode = self.head
                self.head = node
                
            elif node.data < self.tail.data:
                """Insertion at tail"""
                self.tail.nextNode = node
                self.tail = node
                
            else:
                """Insert at any other position"""
                ptr = self.head
                while ptr.nextNode.data > node.data:
                    ptr = ptr.nextNode
            
                node.nextNode = ptr.nextNode
                ptr.nextNode = node
    
    def removeHead(self):
        self.head = self.head.nextNode
        """Garbage collector will remove the node without any references"""


# ## Nearest Neighbour

# In[40]:


class NearestNeighbour:
    
    __dataX = None
    __dataY = None
    __distanceMeasure = None
    
    def __init__(self, x, y, distanceMeasure):
        self.__dataX = x
        self.__dataY = y
        self.__distanceMeasure = distanceMeasure
            
    def getTopKTargets(self, targetId, k):
        targetIndex = np.where(self.__dataY == targetId)
        x = self.__dataX[targetIndex][0]
        distance = np.apply_along_axis(self.__distanceMeasure, 1, self.__dataX, x)
        """Implementing algorithm for finding top k closest nodes. Here instead of storing distance
        I am storing the tuple of distance and target. So no back tracing from distance to target is required
        Furthermore I am using linked list of size K.
        Worst case senerio the insertion will be order of K but removal will be order of 1. So no need to shift the
        values in the list and furthermore no need to keep on sorting it again and again with each insertion.
        The list will be pre sorted. This will further optimize the process of finding the K candidates."""
        
        """Step-1 initialize the linked list with tuple and keep it sorted in descending 
        order this process in in O(K^^2)"""
        
        invList = InvSortedLinkedList()
        iterationIndex = 0
        while iterationIndex < k:
            
            if iterationIndex != targetIndex:
     
                n = Node()
                n.data = distance[iterationIndex]
                n.payload = self.__dataY[iterationIndex]
                invList.insert(n)
                
            iterationIndex+=1
            
        """Step-2 check if any candidate distance is less than largest distance(head)"""
        for i in range(iterationIndex + 1, len(distance)):
            
            if i == targetIndex:
                continue
                
            if distance[i] < invList.head.data:
                n = Node()
                n.data = distance[i]
                n.payload = self.__dataY[i]
                
                """Add the candidate O(k)"""
                invList.insert(n)
                
                """Remove the largest distance(head) from list O(1)"""
                invList.removeHead()
        
        ptr = invList.head
        kTargets = []
        kDistance = []
        while ptr != None:
            kDistance.append(ptr.data)
            kTargets.append(ptr.payload)
            ptr = ptr.nextNode
        
        """Sort the targets from best to worst"""
        kTargets = kTargets[::-1]
        kDistance = kDistance[::-1]
        return kTargets, kDistance


# ## Collaborative Filter

# In[41]:


class collaborativeFilterAlgorithm:
    USERKNN = 1
    ITEMKNN = 2
    """Some linear combinations of both approaches. Needs to be experimented"""
    EXPERIMENTAL = 3 


# In[42]:


class collaborativeFilter:
    
    userKNN = None
    itemKNN = None
    algorithm = None
    ratingsData = None
    
    def __init__(self, metric, algorithm, ratingsData, userX = None, userY = None, itemX = None, itemY = None):
        self.algorithm = algorithm
        self.ratingsData = ratingsData
        
        if self.algorithm == collaborativeFilterAlgorithm.USERKNN:
            self.userKNN = NearestNeighbour(userX, userY, metric)
        elif self.algorithm == collaborativeFilterAlgorithm.ITEMKNN:
            self.itemKNN = NearestNeighbour(itemX, itemsY, metric)
        elif self.algorithm == collaborativeFilterAlgorithm.EXPERIMENTAL:
            self.userKNN = NearestNeighbour(userX, userY, metric)
            self.itemKNN = NearestNeighbour(itemX, itemsY, metric)
    
    
    def predictScore(self, userId, itemId, userK = None, itemK = None):
        
        if self.algorithm == collaborativeFilterAlgorithm.USERKNN:
            
            similarUsers, userDistance = self.userKNN.getTopKTargets(userId, userK)
            userDistance = np.array(userDistance)
            avgUserRating = np.average(self.ratingsData.loc[self.ratingsData.userId == userId].rating.values)
            userSimilarity = 1 - userDistance
            
            userItemRatingList = []
            avgUserRatingList = []
            
            """Get all user ratings"""
            for user in similarUsers:
                
                userRating = self.ratingsData.loc[(self.ratingsData.userId == user)]
                
                avgUserRatingList.append(np.average(userRating.rating.values))
                userRating = userRating[(userRating.itemId == itemId)].rating.values
                
                
                if len(userRating) > 0:
                    userItemRatingList.append(userRating[0])
                else:
                    userItemRatingList.append(0)
                    
            userItemRatingList = np.array(userItemRatingList)
            avgUserRatingList = np.array(avgUserRatingList)
            
            prediction = avgUserRating + ((np.sum(userSimilarity * (userItemRatingList - avgUserRatingList)))/(np.sum(np.abs(userSimilarity))))  
            
            return prediction
        
        elif self.algorithm == collaborativeFilterAlgorithm.ITEMKNN:
            
            similarItems, itemDistance = self.itemKNN.getTopKTargets(itemId, itemK)
            itemDistance = np.array(itemDistance)
            
            avgItemRating = np.average(self.ratingsData.loc[self.ratingsData.itemId == itemId].rating.values)
            itemSimilarity = 1 - itemDistance
            
            
            userItemRatingList = []
            avgItemRatingList = []
            """Get rating for all similar items"""
            for item in similarItems:
                
                itemRating = self.ratingsData.loc[(self.ratingsData.itemId == item)] 
                avgItemRatingList.append(np.average(itemRating.rating.values))
                
                userRating = itemRating[itemRating.userId == userId].rating.values
                
                if len(userRating) > 0:
                    userItemRatingList.append(userRating[0])
                else:
                    userItemRatingList.append(0)
            

            
            userItemRatingList = np.array(userItemRatingList)
            avgItemRatingList = np.array(avgItemRatingList)
            
            prediction = avgItemRating + ((np.sum(itemSimilarity * (userItemRatingList - avgItemRatingList)))/(np.sum(np.abs(itemSimilarity))))
            return prediction
        
        elif self.algorithm == collaborativeFilterAlgorithm.EXPERIMENTAL:
            """Implementation for experimental is pending
            The idea was to use some linear combination of Item and User method.
            Using average for now"""
            
            similarItems, itemDistance = self.itemKNN.getTopKTargets(itemId, itemK)
            itemDistance = np.array(itemDistance)
            
            avgItemRating = np.average(self.ratingsData.loc[self.ratingsData.itemId == itemId].rating.values)
            itemSimilarity = 1 - itemDistance
            
            
            userItemRatingList = []
            avgItemRatingList = []
            """Get rating for all similar items"""
            for item in similarItems:
                
                itemRating = self.ratingsData.loc[(self.ratingsData.itemId == item)] 
                avgItemRatingList.append(np.average(itemRating.rating.values))
                
                userRating = itemRating[itemRating.userId == userId].rating.values
                
                if len(userRating) > 0:
                    userItemRatingList.append(userRating[0])
                else:
                    userItemRatingList.append(0)
            

            
            userItemRatingList = np.array(userItemRatingList)
            avgItemRatingList = np.array(avgItemRatingList)
            
            itemPrediction = avgItemRating + ((np.sum(itemSimilarity * (userItemRatingList - avgItemRatingList)))/(np.sum(np.abs(itemSimilarity))))
            
            similarUsers, userDistance = self.userKNN.getTopKTargets(userId, userK)
            userDistance = np.array(userDistance)
            avgUserRating = np.average(self.ratingsData.loc[self.ratingsData.userId == userId].rating.values)
            userSimilarity = 1 - userDistance
            
            userItemRatingList = []
            avgUserRatingList = []
            
            """Get all user ratings"""
            for user in similarUsers:
                
                userRating = self.ratingsData.loc[(self.ratingsData.userId == user)]
                
                avgUserRatingList.append(np.average(userRating.rating.values))
                userRating = userRating[(userRating.itemId == itemId)].rating.values
                
                
                if len(userRating) > 0:
                    userItemRatingList.append(userRating[0])
                else:
                    userItemRatingList.append(0)
                    
            userItemRatingList = np.array(userItemRatingList)
            avgUserRatingList = np.array(avgUserRatingList)
            
            userPrediction = avgUserRating + ((np.sum(userSimilarity * (userItemRatingList - avgUserRatingList)))/(np.sum(np.abs(userSimilarity))))  
            
            return (itemPrediction + userPrediction)/2


# ## RMSE

# In[43]:


"""Model accuracy estimator RMSE"""

def RMSE(yTrue, yPrediction):
    n = yTrue.shape[0]
    return sqrt((1.0) * np.sum(np.square((yTrue - yPrediction))))/n


# ## Preprocessing

# In[44]:


directoryPath = "Data"


# In[45]:


itemFeatures = ["movieId","movieTitle","releaseDate",
                "videoReleaseDate","IMDBURL","unknown",
                "Action","Adventure","Animation",
                "Children's","Comedy","Crime","Documentary",
                "Drama","Fantasy","Film-Noir","Horror",
                "Musical","Mystery","Romance","Sci-Fi",
                "Thriller","War","Western"]

itemsData = pd.read_csv(directoryPath+"/ml-100k/u.item", names = itemFeatures, sep="|")
itemsData.head()


# In[46]:


usersData = pd.read_csv(directoryPath+"/ml-100k/u.user", names = ["userId", 
                                                            "age", "gender", "occupation", "zipcode"], sep="|")
usersData.head()


# In[47]:


"""First we will drop unrequired features"""

"""From user we need userId, age, gender, occupation"""
userFeatures = ["age", "gender", "occupation"]

"""From items we will drop IMDB URL, video release date, movie title. Movie title seems bit 
important and can be used later. But for now we will only focus on latent features
"""

itemFeatures.remove("IMDBURL")
itemFeatures.remove("videoReleaseDate")
itemFeatures.remove("movieTitle")
itemFeatures.remove("movieId")


# In[48]:


"""Preprocessing user data"""
usersY = usersData['userId']
usersData = usersData[userFeatures]
usersData = pd.get_dummies(usersData)
usersData.head()
usersX = usersData.as_matrix()


# In[49]:


"""Preprocessing user data"""
itemsY = itemsData["movieId"]
itemsData = itemsData[itemFeatures]
itemsData['releaseDate'] = pd.to_datetime(itemsData['releaseDate']).dt.year.values
itemsData = pd.get_dummies(itemsData)
itemsData.head()
itemsX = itemsData.as_matrix()


# ## Kfold validation

# In[50]:


gridK = [2, 3, 4, 5, 6, 7, 8, 9, 10]
folds = []

u1Base = pd.read_csv(directoryPath+"/ml-100k/u1.base", names = ["userId", 
                                                            "itemId", "rating", "timestamp"], sep="\t")
folds.append(u1Base)
u2Base = pd.read_csv(directoryPath+"/ml-100k/u2.base", names = ["userId", 
                                                            "itemId", "rating", "timestamp"], sep="\t")
folds.append(u2Base)
u3Base = pd.read_csv(directoryPath+"/ml-100k/u3.base", names = ["userId", 
                                                            "itemId", "rating", "timestamp"], sep="\t")
folds.append(u3Base)
u4Base = pd.read_csv(directoryPath+"/ml-100k/u4.base", names = ["userId", 
                                                            "itemId", "rating", "timestamp"], sep="\t")
folds.append(u4Base)
u5Base = pd.read_csv(directoryPath+"/ml-100k/u5.base", names = ["userId", 
                                                            "itemId", "rating", "timestamp"], sep="\t")
folds.append(u5Base)
indices = range(0, len(folds))


# In[ ]:


RMSEListUserMethod = []
RMSEListItemMethod = []

for k in gridK:
    foldRMSE = [] 
    for i in indices:
        
        validationSet = folds[i]
        trainSet = []
        yPredictionUserMethod = []
        yPredictionItemMethod = []
        
        
        for j in indices:    
            if j != i:
                trainSet.append(folds[j])
        trainSet = pd.concat(trainSet)
        
        userMethodCF = collaborativeFilter(DistanceMeasure.CosineDistance, collaborativeFilterAlgorithm.USERKNN,
                                          trainSet, usersX, usersY.values, itemsX, itemsY.values)
        
        yActual = np.array(validationSet.rating.values)
        userTest = validationSet.userId.values
        itemTest = validationSet.itemId.values
        for user, item in zip(userTest, itemTest):
            yPredictionUserMethod.append(userMethodCF.predictScore(user, item, userK = k))
        
        foldRMSE.append(yActual, yPredictionUserMethod)
    print np.average(foldRMSE)
    RMSEListUserMethod.append(np.average(foldRMSE))
        


# Not able to run KFold on system. Takes lot of time.
