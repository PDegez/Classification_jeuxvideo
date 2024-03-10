import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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


def knn_model_cross_val(matrix, classes):
    knn_classifier = KNeighborsClassifier(n_neighbors=28)
    y_pred = cross_val_predict(knn_classifier, matrix, classes, cv=10)
    accuracy = accuracy_score(classes, y_pred)
    conf_matrix = confusion_matrix(classes, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot()
    plt.show()
    return accuracy


def find_best_k(matrix, classes):
    k_values = [i for i in range(1, 44)]
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, matrix, classes, cv=5)
        scores.append(np.mean(score))
    plt.figure(figsize=(12, 7))
    plt.plot(k_values, scores, marker="o", linestyle="-")
    plt.title("Accuracy depending on the number of neighbors (k)")
    plt.xlabel("Number of neighbors (k)")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()


def knn_model_split_train(matrix, classes):
    x_train, x_test, y_train, y_test = train_test_split(
        matrix, classes, test_size=0.2, random_state=42
    )
    knn_classifier = KNeighborsClassifier(n_neighbors=9)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=knn_classifier.classes_)
    disp.plot()
    plt.show()
    return accuracy


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 visualization.py corpus/")
    corpus = glob.glob(f"{sys.argv[1]}/*/*.txt")
    matrix = get_matrix(corpus)
    classes = [path.split("/")[-2] for path in corpus]
    if len(sys.argv) == 3 and sys.argv[2] == "split":
        accuracy = knn_model_split_train(matrix, classes)
    else:
        accuracy = knn_model_cross_val(matrix, classes)
    # find_best_k(matrix, classes)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
