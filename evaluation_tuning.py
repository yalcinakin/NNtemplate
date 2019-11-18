######################################
################## Evaluation

#Evaluate ANN with 10-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # Add Classifier (same with training)
    # DO NOT include classifier.fit( ...)

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
    # Add Classifier (same with training)
    # DO NOT include classifier.fit( ...)
    
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