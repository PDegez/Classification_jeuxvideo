import sys
import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_corpus(corpus):
    content = []
    for filename in corpus:
        with open(filename, "r") as file:
            content.append(file.read())
    return content


def plot_vectors(output):
    plt.figure()
    classes = output["Class"].unique()
    for class_name in classes:
        if class_name == "Fantasy":
            color = "maroon"
        elif class_name == "Family":
            color = "moccasin"
        elif class_name == "Comedy":
            #color = "tan"
            color = "lightcoral"
        elif class_name == "Sci-Fi":
            color = "peru"
        else:
            color = "black"
        class_data = output[output["Class"] == class_name]
        plt.scatter(
            class_data["1"], class_data["2"], label=class_name, color=color, alpha=0.75
        )
    plt.legend()
    plt.grid()
    plt.title("PCA reduction 2 components")
    plt.show()


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 visualization.py corpus/")
    corpus = glob.glob(f"{sys.argv[1]}/*/*.txt")
    classes = [os.path.basename(os.path.dirname(file_path)) for file_path in corpus]
    content = load_corpus(corpus)
    # vectorizer = CountVectorizer(input="content", stop_words="english")
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(content).toarray()
    df = pd.DataFrame(matrix, columns=vectorizer.get_feature_names_out())
    # print(df)
    pca = PCA(n_components=2)
    pca_matrix = pca.fit_transform(matrix)
    print(pca.explained_variance_ratio_)
    output = pd.DataFrame(pca_matrix, columns=["1", "2"])
    output["Class"] = classes
    plot_vectors(output)


if __name__ == "__main__":
    main()
