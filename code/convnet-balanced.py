
from tensorflow import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from numpy import number
from numpy import fliplr
from numpy import rot90
from numpy import asarray
from numpy import apply_along_axis
from pandas import read_csv
import numpy as np
import time



# import code pulled from the notebook section where the data was acquired
# https://www.kaggle.com/ashwani07/emnist-using-keras-cnn
train = read_csv("./EMNISTcsv/emnist-balanced-train.csv",delimiter = ',')
test = read_csv("./EMNISTcsv/emnist-balanced-test.csv", delimiter = ',')

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
print ("train_X:",train_X.shape)

test_X = asarray(test_X)
test_X = apply_along_axis(rotate, 1, test_X)
print ("test_X:",test_X.shape)
print("type ", type(test_X[0][0][0]))
print('Training data shape : ', train_X.shape, train_Y.shape)

features = []
containers = []

for nimage in range(len(train_X)):
    features.append(train_X[nimage])
    containers.append(train_Y[nimage])

for nimage in range(len(test_X)):
    features.append(test_X[nimage])
    containers.append(test_Y[nimage])

features_arr = np.array(features, 'float64')


train_X = features_arr.reshape(-1, 28,28, 1)
#test_X = test_X.reshape(-1, 28,28, 1)
train_X = train_X.astype('float32')
#test_X = test_X.astype('float32')
train_X = train_X / 255.
#test_X = test_X / 255.
train_Y_one_hot = to_categorical(containers)
#test_Y_one_hot = to_categorical(test_Y)

# combining training and testing to use sklearn's split rather than the dataset's

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

# convolutional neural network code below mostly from the following guide
# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

batch_size = 32
epochs = 10
num_classes = 47

number_model = Sequential()
number_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
number_model.add(LeakyReLU(alpha=0.1))
number_model.add(MaxPooling2D((2, 2),padding='same'))
number_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
number_model.add(LeakyReLU(alpha=0.1))
number_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
number_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
number_model.add(LeakyReLU(alpha=0.1))                  
number_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
number_model.add(Flatten())
number_model.add(Dense(128, activation='linear'))
number_model.add(LeakyReLU(alpha=0.1))                  
number_model.add(Dense(num_classes, activation='softmax'))

number_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# number_model.summary()

# train
start = time.time()
number_train = number_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
end_train = time.time()

score, acc = number_model.evaluate(valid_X, valid_label, batch_size=batch_size)
end_test = time.time()
print("Training time: ", end_train - start)
print("Test time: ", end_test - end_train)
print('Test score:', score)
print('Test accuracy:', acc)

