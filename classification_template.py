# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######################################
################## PreProcessing

# Importing the dataset
dataset = pd.read_csv('sample.csv')
X = dataset.iloc[:, [2, 3]].values  #### Change Indexes!!
y = dataset.iloc[:, -1].values

# Encode data from categorical_data, if necessary

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


######################################
################## Classification

# Fitting classifier to the Training set
# Add Classifier





######################################
################## Prediction

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


######################################
######################################
################## OPTIONAL PARTS

######################################
################## Visualization

# Add visualization to visualize training and test sets from visualization template, if necessary


######################################
################## Evaluation

# Add evaluation from evaluation_tuning.py, to evaluate the classifier, if necessary


######################################
################## Tuning hyperparameters

# Add tuning from evaluation_tuning.py, to tune hyperparameters, if necessary

