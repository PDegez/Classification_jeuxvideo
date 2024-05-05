import sys
import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_corpus(corpus):
    content = []
    for filename in corpus:
        with open(filename, "r") as file:
            content.append(file.read())
    return content


def calculate_PCA(content):
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(content).toarray()
    pca = PCA()
    pca_matrix = pca.fit_transform(matrix)
    exp_var_ratio = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_ratio)
    return cum_sum_eigenvalues


def plot_var_ratio(cum_sum_eigenvalues):
    plt.step(
        range(0, len(cum_sum_eigenvalues)),
        cum_sum_eigenvalues,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 PCA_explained_variance.py corpus/")
    corpus = glob.glob(f"{sys.argv[1]}/*/*.txt")
    content = load_corpus(corpus)
    cum_sum_eigenvalues = calculate_PCA(content)
    plot_var_ratio(cum_sum_eigenvalues)


if __name__ == "__main__":
    main()
