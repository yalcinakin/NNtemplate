# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:15:44 2019

@author: Yalcin Akin
"""

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

######################################
################## PreProcessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],    # The column numbers to be transformed
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Deleting dummy variable
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

######################################
################## Classification

# Fitting classifier to the Training set
# ANN Classifier

# Importing the libraries
#import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout  ## To add Dropout Regularization

# ANN
classifier = Sequential()  #initialize
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim=11))  # first hidden layer and input layer
#classifier.add(Dropout(p= 0.1))

classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))  # second hidden layer
#classifier.add(Dropout(p= 0.1))

classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))  # output layer

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=100, batch_size=10)

######################################
################## Prediction

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## single prediction
#xvar = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])  # example given data
#xvar = sc.transform(xvar)
#pred_single = classifier.predict(xvar)
#pred_single = (pred_single > 0.5)

######################################
################## Evaluation

#Evaluate ANN with 10-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()  #initialize
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim=11))  # first hidden layer and input layer
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))  # second hidden layer
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))  # output layer
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10)
#accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10, n_jobs=-1 )

mean = accuracies.mean()
variance = accuracies.std()

######################################
################## Tuning Hyperparameters

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()  #initialize
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim=11))  # first hidden layer and input layer
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))  # second hidden layer
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))  # output layer
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
parameters = {
        'batch_size': [25, 32],
        'epochs': [100, 500],
        'optimizer': ['adam', 'rmsprop']
        }

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, cv = 10, scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_