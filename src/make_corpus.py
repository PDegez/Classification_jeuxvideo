import sys
import os
import csv

def load_metadata(path):
    return [*csv.DictReader(open(path))]

def main():
    metadata=load_metadata("aux/imdb_video_games.csv")
    for i, row in enumerate(metadata):
        genre = row["Genre"].replace(", ", "_")
        title = row["Title"].replace(" ", "_").replace("/", "_")
        key = row["Popularity"]
        content = row["Summary"]
        if content != "":

            if os.path.isdir(f"corpus/{genre}"):
                with open(f"corpus/{genre}/{key}_{title}.txt", "w") as file:
                    file.write(content)
            else :
                os.mkdir(f"corpus/{genre}")
                with open(f"corpus/{genre}/{key}_{title}.txt", "w") as file:
                    file.write(content)
        


if __name__ == "__main__":
    main()
