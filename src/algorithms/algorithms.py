#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:18:37 2024

@author: pauline
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def knn_classification(matrix, classes):
    knn_classifier = KNeighborsClassifier(n_neighbors=29)
    y_pred = cross_val_predict(knn_classifier, matrix, classes, cv=10)
    return y_pred


def naive_bayes_classification(matrix, classes):
    nb_classifier = MultinomialNB()
    y_pred = cross_val_predict(nb_classifier, matrix, classes, cv=10)
    return y_pred


def svm_classification(matrix, classes):
    svm_classifier = SVC(kernel='linear')
    y_pred = cross_val_predict(svm_classifier, matrix, classes, cv=10)
    return y_pred


def random_forest_classification(matrix, classes):
    rf_classifier = RandomForestClassifier()
    y_pred = cross_val_predict(rf_classifier, matrix, classes, cv=10)
    return y_pred


def decision_tree_classification(matrix, classes):  
    dt_classifier = DecisionTreeClassifier()
    y_pred = cross_val_predict(dt_classifier, matrix, classes, cv=10)
    return y_pred


