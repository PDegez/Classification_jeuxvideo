import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import glob


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

def naive_bayes_classification(matrix, classes):
    nb_classifier = MultinomialNB()
    y_pred = cross_val_predict(nb_classifier, matrix, classes, cv=10)
    accuracy = accuracy_score(classes, y_pred)
    precision = precision_score(classes, y_pred, average="macro")
    recall = recall_score(classes, y_pred, average="macro")
    f_score = f1_score(classes, y_pred, average="macro")
    conf_matrix = confusion_matrix(classes, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()
    return accuracy, precision, recall, f_score


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 visualization.py corpus/")
    corpus = glob.glob(f"{sys.argv[1]}/*/*.txt")
    matrix = get_matrix(corpus)
    classes = [path.split("/")[-2] for path in corpus]
    accuracy, precision, recall, f_score = naive_bayes_classification(matrix, classes)
    print(f"Accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f_score: {f_score}")


if __name__ == "__main__":
    main()
