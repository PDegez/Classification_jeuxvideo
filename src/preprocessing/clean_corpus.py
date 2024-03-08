import glob

CATEGORIES = ["adventure", "comedy", "family", "fantasy", "sci-fi"]


def main():
    all_files = glob.glob("corpus/*/*.txt")
    for path in all_files:
        with open(path, "r") as file:
            content = file.read()
        for category in CATEGORIES:
            content = content.replace(category, "")
        with open(path, "w") as file:
            file.write(content)

main()
