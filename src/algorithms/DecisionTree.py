#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 23:42:47 2024

@author: pauline
"""


from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier #, plot_tree
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import os
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, recall_score
import argparse


# pillage des fonctions de Valentine
def load_corpus(corpus):
    content = []
    for filename in corpus:
        with open(filename, "r") as file:
            content.append(file.read())
    return content

# vectorisation (tdidf = réduction des dimensions)
def get_matrix(corpus):
    content = load_corpus(corpus)
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(content).toarray()
    return matrix

def dt_model_cross_val(matrix, classes):
    
    # hyperparamètres adaptés en tatonnant
    # dt_classifier = DecisionTreeClassifier(max_depth=40, max_leaf_nodes=37, min_samples_leaf=20)
    dt_classifier = DecisionTreeClassifier()
    y_pred = cross_val_predict(dt_classifier, matrix, classes, cv=10)
    
    # Print des scores
    accuracy = accuracy_score(classes, y_pred)
    precision = precision_score(classes, y_pred, average="macro")
    recall = recall_score(classes, y_pred, average="macro")
    f_score = f1_score(classes, y_pred, average="macro")
    print(f"accuracy : {accuracy}")
    print(f"precision : {precision}")
    print(f"recall : {recall}")
    print(f"f_score : {f_score}")
    
    conf_matrix = confusion_matrix(classes, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()
    return print ("cross_val done")

def dt_model_split_train(matrix, classes):
    # division du corpus en partie test vs partie train
    X_train, X_test, y_train, y_test = train_test_split(matrix, classes, test_size=0.2, random_state=42)

    # Création de l'arbre : hyperparamètres déduits en tatonnant
    # min_samples_split=20, max_depth=35, max_leaf_nodes=37, min_samples_leaf=20
    clf = DecisionTreeClassifier(max_depth=40, max_leaf_nodes=37, min_samples_leaf=20)
    clf.fit(X_train, y_train)

    # Print des scores
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot la matrice de confusion : labels réels vs les labels prédits
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()
    
    return print("split_train done")


def main():

    corpus_global = glob.glob(f"{args.input_directory}/*/*.txt")
    corpus = [chemin for chemin in corpus_global 
              if not chemin.startswith(os.path.join(args.input_directory,
                                                    "Adventure"))]
    matrix = get_matrix(corpus)
    classes = [path.split("/")[-2] for path in corpus]

    if not args.cross_val:
        dt_model_split_train(matrix, classes)
        
    if args.cross_val:
        dt_model_cross_val(matrix, classes)
    
    return print("fin du script")


# Plot : (ne pas afficher tant que pas compris parce que JESUS)
#plt.figure(figsize=(10, 8))
#plot_tree(clf, filled=True, class_names=classes)
#plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification : decision tree")
    parser.add_argument("-c", "--cross_val",
                        action="store_true", help="cross validation")
    parser.add_argument("input_directory",
                        help="Chemin vers le dossier contenant le corpus") 
    args = parser.parse_args()
    main()

