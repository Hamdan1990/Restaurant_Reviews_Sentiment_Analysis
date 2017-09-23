# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:22:20 2017

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

#Fitting the Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
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

#Calculating Precision
from sklearn.metrics import precision_score
Precision = precision_score(y_test, y_pred)
print("Precision : %0.2f" %Precision)

#Calculating Recall
from sklearn.metrics import recall_score
Recall = recall_score(y_test, y_pred)
print("Recall : %0.2f " %Recall)

#Calculating F1 Score
from sklearn.metrics import f1_score
F1 = f1_score(y_test, y_pred)
print("F1 Score: %0.2f " %F1)