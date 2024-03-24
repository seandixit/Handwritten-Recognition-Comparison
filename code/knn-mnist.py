import numpy as np
from skimage.feature import hog
from skimage.transform import rescale
from skimage.io import imshow, imread
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.util import invert
import joblib
from matplotlib import pyplot as plt
import os
from numpy import number
from numpy import fliplr
from numpy import rot90
from numpy import asarray
from numpy import apply_along_axis
from pandas import read_csv
import time

#heavily guided by this website: https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea

project_dir = os.path.dirname(os.path.abspath(__file__))


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

def rotate(dataset):
    dataset = dataset.reshape([28, 28])
    dataset = fliplr(dataset)
    dataset = rot90(dataset)
    return dataset

# Flip and rotate 
train_X = asarray(train_X)
train_X = apply_along_axis(rotate, 1, train_X)
#print ("train_X:",train_X.shape)

test_X = asarray(test_X)
test_X = apply_along_axis(rotate, 1, test_X)
#print ("test_X:",test_X.shape)
#print('Training data shape : ', train_X.shape, train_Y.shape)

#****end of import code

# combining datasets and applying HOG for feature extraction
features = []
containers = []

for nimage in range(len(train_X)):
    features.append(hog(train_X[nimage], orientations=8, pixels_per_cell=(7,7), cells_per_block=(4, 4)))
    containers.append(train_Y[nimage])

for nimage in range(len(test_X)):
    features.append(hog(test_X[nimage], orientations=8, pixels_per_cell=(7,7), cells_per_block=(4, 4)))
    containers.append(test_Y[nimage])


#convert to numpy array for use with KNeighborsClassifier
features_arr = np.array(features, 'float64')
X_train, X_test, y_train, y_test = train_test_split(features_arr, containers)

# train
start = time.time()

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
end_train = time.time()
model_score = knn.score(X_test, y_test)
end_test = time.time()


print("Training time: ", end_train - start)
print("Test time: ", end_test - end_train)
print("Accuracy: ", model_score)



# save model
#joblib.dump(knn, os.path.join(project_dir, 'models', 'knn_model.pkl'))
