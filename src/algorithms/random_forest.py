#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:41:59 2024

@author: pauline
"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def load_corpus(corpus):
    content = []
    for filename in corpus:
        with open(filename, "r") as file:
            content.append(file.read())
    return content


def get_matrix(corpus):
    content = load_corpus(corpus)
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(content).toarray()
    return matrix


def random_forest_classification(matrix, classes):
    rf_classifier = RandomForestClassifier()
    y_pred = cross_val_predict(rf_classifier, matrix, classes, cv=10)
    return y_pred


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 visualization.py corpus/")
    corpus = glob.glob(f"{sys.argv[1]}/*/*.txt")
    matrix = get_matrix(corpus)
    classes = [path.split("/")[-2] for path in corpus]
    prediction = random_forest_classification(matrix, classes)
    print(classification_report(classes, prediction))
    conf_matrix = confusion_matrix(classes, prediction)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()



if __name__ == "__main__":
    main()

