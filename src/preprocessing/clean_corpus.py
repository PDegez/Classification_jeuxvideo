import glob
import re
import spacy

CATEGORIES = ["adventure", "comedy", "family", "fantasy", "sci-fi"]


def tokenize(text: str) -> list[str]:
    return [
        token.replace("’", "'")
        for token in re.findall(r"\b\w+?\b(?:'|’)?", text)
    ]

def lemmatize(text: str) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas


def main():
    all_files = glob.glob("corpus/*/*.txt")
    for path in all_files:
        with open(path, "r") as file:
            content = file.read()
        tokens = lemmatize(content)
        with open(path, "w") as file:
            for token in tokens:
                if token not in CATEGORIES:
                    file.write(token.lower() + " ")


main()
