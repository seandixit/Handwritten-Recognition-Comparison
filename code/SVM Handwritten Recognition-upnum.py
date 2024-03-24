#!/usr/bin/env python
# coding: utf-8

# In[110]:

#***
# Code influenced by: https://youtu.be/7sz4WpkUIIs

import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from skimage.feature import hog
from skimage.transform import rescale
from skimage.io import imshow, imread
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.util import invert
import joblib
from matplotlib import pyplot as plt
from numpy import number
from numpy import fliplr
from numpy import rot90
from numpy import asarray
from numpy import apply_along_axis
from pandas import read_csv
import time

project_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(project_dir, 'archive', 'Img', '')
# import code pulled from the notebook section where the data was acquired
# https://www.kaggle.com/ashwani07/emnist-using-keras-cnn
train = read_csv("./EMNISTcsv/emnist-mnist-train.csv",delimiter = ',')
test = read_csv("./EMNISTcsv/emnist-mnist-test.csv", delimiter = ',')

train_X = train.iloc[:,1:]
train_Y = train.iloc[:,0]
del train
test_X = test.iloc[:,1:]
test_Y = test.iloc[:,0]
del test

train = read_csv("./EMNISTcsv/emnist-letters-train.csv",delimiter = ',')
test = read_csv("./EMNISTcsv/emnist-letters-test.csv", delimiter = ',')

train_X_letter = train.iloc[:,1:]
train_Y_letter = train.iloc[:,0]
del train
test_X_letter = test.iloc[:,1:]
test_Y_letter = test.iloc[:,0]
del test

def rotate(dataset):
    dataset = dataset.reshape([28, 28])
    dataset = fliplr(dataset)
    dataset = rot90(dataset)
    return dataset

# Flip and rotate 
train_X = asarray(train_X)
train_X = apply_along_axis(rotate, 1, train_X)
print ("train_X:",train_X.shape)

test_X = asarray(test_X)
test_X = apply_along_axis(rotate, 1, test_X)
print ("test_X:",test_X.shape)

train_X_letter = asarray(train_X_letter)
train_X_letter = apply_along_axis(rotate, 1, train_X_letter)
print ("train_X:",train_X_letter.shape)

test_X_letter = asarray(test_X_letter)
test_X_letter = apply_along_axis(rotate, 1, test_X_letter)
print ("test_X:",test_X_letter.shape)
#****end of import code


# In[ ]:


# In[156]:


# combining datasets and applying HOG for feature extraction
features = []
containers = []

for nimage in range(len(train_X)):
    features.append(hog(train_X[nimage], orientations=8, pixels_per_cell=(7,7), cells_per_block=(4, 4)))
    containers.append(train_Y[nimage])

for nimage in range(len(test_X)):
    features.append(hog(test_X[nimage], orientations=8, pixels_per_cell=(7,7), cells_per_block=(4, 4)))
    containers.append(test_Y[nimage])

for nimage in range(len(train_X_letter)):
    features.append(hog(train_X_letter[nimage], orientations=8, pixels_per_cell=(7,7), cells_per_block=(4, 4)))
    containers.append(train_Y_letter[nimage]+9)

for nimage in range(len(test_X_letter)):
    features.append(hog(test_X_letter[nimage], orientations=8, pixels_per_cell=(7,7), cells_per_block=(4, 4)))
    containers.append(test_Y_letter[nimage]+9)


#convert to numpy array for use with KNeighborsClassifier
features_arr = np.array(features, 'float64')
X_train, X_test, y_train, y_test = train_test_split(features_arr, containers)


# In[157]:


# training model
from sklearn import svm
model_linear = svm.SVC(kernel='linear', degree=3, gamma='scale')

print("starting linear test...")
start = time.time()

model_linear.fit(X_train, y_train)
end_train = time.time()
print("Training time: ", end_train - start)

# making predition
#y_pred = model_linear.predict(X_test)

# evaluating model for accuracy
linear_score = model_linear.score(X_test,y_test)
end_test = time.time()

print("Test time: ", end_test - end_train)
print("Accuracy: ", linear_score)


# In[158]:


# model with kernel RBF
model_RBF = svm.SVC(degree=3, gamma='scale', kernel='rbf')

print("starting RBF test...")
start = time.time()
model_RBF.fit(X_train, y_train)
end_train = time.time()
print("Training time: ", end_train - start)

# making prediction
rbf_score = model_RBF.score(X_test, y_test)
end_test = time.time()
print("Test time: ", end_test - end_train)
print("Accuracy: ", rbf_score)


# In[159]:


#from sklearn.metrics import classification_report 
#predictions = model_linear.predict(X_test)
#print(classification_report(y_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:




