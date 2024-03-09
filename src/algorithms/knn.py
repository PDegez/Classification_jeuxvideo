import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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


def knn_model(matrix, classes):
    x_train, x_test, y_train, y_test = train_test_split(matrix, classes, test_size=0.2, random_state=42)
    knn_classifier = KNeighborsClassifier(n_neighbors=9)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 visualization.py corpus/")
    corpus = glob.glob(f"{sys.argv[1]}/*/*.txt")
    matrix = get_matrix(corpus)
    classes = [path.split("/")[-2] for path in corpus]
    accuracy = knn_model(matrix, classes)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
