from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import json

# This file converts a particular Dataset into a file with a sentence in each line.
# A file with a sentence in each line will be the input to the next phase: preprocessing.

DATA_FILE = "data/cord19-filtered.json"
OUT_FILE = "data/cord19-standard.txt"

if __name__ == "__main__":
    buffer = []
    detokenizer = TreebankWordDetokenizer()
    with open(DATA_FILE) as f_json:
        data = json.load(f_json)
    with open(OUT_FILE, "w", encoding="utf-8") as out_file:
        for article in tqdm(data):
            title: str = article["title"]
            abstract: str = article["abstract"]
            texts: list = article["text"]
            out_file.write(detokenizer.detokenize(title.split(" ")) + "\n")
            out_file.write(detokenizer.detokenize(abstract.split(" ")) + "\n")
            for text in texts:
                out_file.write(detokenizer.detokenize(text.split(" ")) + "\n")
            out_file.write(title)
