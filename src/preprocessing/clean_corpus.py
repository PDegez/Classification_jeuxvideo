import glob
import re

CATEGORIES = ["adventure", "comedy", "family", "fantasy", "sci-fi"]


def tokenize(text: str) -> list[str]:
    return [
        token.replace("’", "'")
        for token in re.findall(r"\b\w+?\b(?:'|’)?", text)
    ]


def main():
    all_files = glob.glob("corpus/*/*.txt")
    for path in all_files:
        with open(path, "r") as file:
            content = file.read()
        tokens = tokenize(content)
        with open(path, "w") as file:
            for token in tokens:
                if token not in CATEGORIES:
                    file.write(token.lower() + " ")


main()
