#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:17:20 2024

@author: Valentine et Pauline
"""

import argparse
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt
from algorithms import (
    knn_classification, 
    naive_bayes_classification,
    decision_tree_classification,
    random_forest_classification,
    svm_classification)


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


def print_error():
    print("Choississez un classifieur : option -c suivi du modele choisi :")
    print("-c NB \tpour lancer Naive Bayes")
    print("-c SVM \tpour lancer SVM")
    print("-c DT \tpour lancer Decision Tree")
    print("-c RF \tpour lancer Random Forest")


# Dictionnaire pour lancer le classifieur en fonction de l'option choisie
modeles = {
    "KNN": knn_classification,
    "NB": naive_bayes_classification,
    "SVM": svm_classification,
    "DT": decision_tree_classification,
    "RF": random_forest_classification, 
}


def save_prediction(prediction, classes, output):
    report = str(classification_report(classes, prediction))
    with open(output, "w") as file:
        file.write(report)


def generate_confusion_matrix(prediction, classes):
    conf_matrix = confusion_matrix(classes, prediction)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()


################################ MAIN ####################################


def main():
    parser = argparse.ArgumentParser(description="lanceur de classifieurs")
    parser.add_argument(
        "input_directory", type=str, help="Chemin vers le dossier contenant le corpus"
    )
    parser.add_argument(
        "-c",
        "--classifieur",
        choices=["KNN", "NB", "SVM", "DT", "RF"],
        help="Choix du modèle de classifieur",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Chemin vers le fichier pour sauvegarder les scores.",
    )
    args = parser.parse_args()
    if not args.classifieur:
        print_error()
        return 1
    output = args.output_file
    if not output:
        output = f"classification-report-{args.classifieur}.txt"
    # Load le corpus et extrait les vecteurs (matrices) et les classes
    corpus = glob.glob(f"{args.input_directory}/*/*.txt")
    classes = [path.split("/")[-2] for path in corpus]
    matrix = get_matrix(corpus)
    
    # Lancement du classifieur
    prediction = modeles[args.classifieur](matrix, classes)
    save_prediction(prediction, classes, output)
    generate_confusion_matrix(prediction, classes)


if __name__ == "__main__":
    main()
