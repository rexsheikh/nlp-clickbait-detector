import random
import nltk
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from nltk import bigrams

import csv

# constant path for kaggle dataset
KAGGLE_DATASET_PATH = "/Users/rexsheikh/Documents/boulder-fall-2025/nlp/clickbait-detector/data/kaggle_clickbait.csv"


# Load the Kaggle clickbait dataset CSV and return counts for news and clickbait headlines
def load_data(path: str = KAGGLE_DATASET_PATH):
    clickbait_1 = 0
    news_0 = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        # Handle potential BOM in first header cell
        if reader.fieldnames:
            reader.fieldnames = [
                fn.lstrip("\ufeff") if isinstance(fn, str) else fn
                for fn in reader.fieldnames
            ]
        for row in reader:
            val = row.get("clickbait")
            if val is None:
                continue
            try:
                label = int(val)
            except ValueError:
                continue
            if label == 1:
                clickbait_1 += 1
            elif label == 0:
                news_0 += 1
    return clickbait_1, news_0


if __name__ == "__main__":
    c1, c0 = load_data()
    print(f"clickbait (label=1): {c1}")
    print(f"news      (label=0): {c0}")
