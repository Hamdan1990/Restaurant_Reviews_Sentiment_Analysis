# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:03:15 2017

@author: Hamdan
"""


import pandas as pd
import numpy as np

#Loading the Cleaned Dataset
dataset = pd.read_csv("CleanedReviews.tsv", sep = '\t', quoting =3)
dataset.drop("Unnamed: 0", axis = 1, inplace=True)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(dataset['Review'].values.astype('U')).toarray()
y = dataset.iloc[:, 0].values

#Dividing into Test and Test Sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Predicitng the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Applyin k-fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv= 10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))