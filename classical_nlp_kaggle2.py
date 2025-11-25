import random
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score
import re
from collections import Counter

# constant path for kaggle dataset
KAGGLE_DATASET_PATH = "/Users/rexsheikh/Documents/boulder-fall-2025/nlp/nlp-clickbait-detector/data/kaggle_clickbait.csv"


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


# ================= Kaggle Naive Bayes + MultinomialNB =================

TOP_N_KAGGLE = 2000

def regex_tokenize(text: str):
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())

def load_kaggle_texts_labels(path: str = KAGGLE_DATASET_PATH):
    texts, labels = [], []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [
                fn.lstrip("\ufeff") if isinstance(fn, str) else fn
                for fn in reader.fieldnames
            ]
        for row in reader:
            headline = row.get("headline")
            lb = row.get("clickbait")
            if headline is None or lb is None:
                continue
            try:
                y = int(lb)
            except ValueError:
                continue
            texts.append(headline)
            labels.append(y)
    return texts, labels

def build_vocab_kaggle(texts, top_n=TOP_N_KAGGLE):
    cnt = Counter()
    for t in texts:
        cnt.update(regex_tokenize(t))
    return [w for w, _ in cnt.most_common(top_n)]

def document_features_kaggle(tokens, vocab):
    token_set = set(tokens)
    return {f"contains({w})": (w in token_set) for w in vocab}

def run_kaggle_experiments():
    print("\n=== Kaggle Clickbait Dataset ===")
    X_texts, y = load_kaggle_texts_labels()
    if not X_texts or not y:
        print("No data loaded from Kaggle file; skipping experiments.")
        return


    def _stratified_split(X, y, test_size=0.2, random_state=42):
        rnd = random.Random(random_state)
        by_class = {}
        for i, label in enumerate(y):
            by_class.setdefault(label, []).append(i)
        train_idx, test_idx = [], []
        for _, idxs in by_class.items():
            rnd.shuffle(idxs)
            n_test = max(1, int(len(idxs) * test_size)) if len(idxs) > 0 else 0
            test_idx.extend(idxs[:n_test])
            train_idx.extend(idxs[n_test:])
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        return X_train, X_test, y_train, y_test

    try:
        from sklearn.model_selection import train_test_split as sk_split
        X_train, X_test, y_train, y_test = sk_split(
            X_texts, y, test_size=0.2, stratify=y, random_state=42
        )
    except Exception:
        X_train, X_test, y_train, y_test = _stratified_split(X_texts, y, test_size=0.2, random_state=42)

    # NLTK Naive Bayes for top-N vocab
    vocab = build_vocab_kaggle(X_train)
    def featurize(texts):
        return [document_features_kaggle(regex_tokenize(t), vocab) for t in texts]

    train_set_k = list(zip(featurize(X_train), y_train))
    test_feats_k = featurize(X_test)

    # Train and evaluate using nltk NaiveBayes
    try:
        from nltk.classify import NaiveBayesClassifier as NLTKNaiveBayes
        nb_k = NLTKNaiveBayes.train(train_set_k)
        pred_nb = [nb_k.classify(x) for x in test_feats_k]
        print(
            f"[NLTK NaiveBayes] Acc={accuracy_score(y_test, pred_nb):.3f}  "
            f"Prec(pos)={precision_score(y_test, pred_nb, pos_label=1):.3f}  "
            f"Rec(pos)={recall_score(y_test, pred_nb, pos_label=1):.3f}"
        )
    except Exception as e:
        print(f"[NLTK NaiveBayes] Error: {e}")

    # sklearn MultinomialNB with CountVectorizer baseline
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline
        mnb_pipeline = make_pipeline(
            CountVectorizer(max_features=TOP_N_KAGGLE),
            MultinomialNB()
        )
        mnb_pipeline.fit(X_train, y_train)
        pred_mnb = mnb_pipeline.predict(X_test)
        # Ensure list for our metric helper
        preds = pred_mnb.tolist() if hasattr(pred_mnb, "tolist") else pred_mnb
        print(
            f"[MultinomialNB]   Acc={accuracy_score(y_test, preds):.3f}  "
            f"Prec(pos)={precision_score(y_test, preds, pos_label=1):.3f}  "
            f"Rec(pos)={recall_score(y_test, preds, pos_label=1):.3f}"
        )
    except Exception as e:
        print(f"[MultinomialNB] Error: {e}")


if __name__ == "__main__":
    c1, c0 = load_data()
    print(f"clickbait (label=1): {c1}")
    print(f"news      (label=0): {c0}")
    # Run Kaggle NB and MultinomialNB comparisons
    run_kaggle_experiments()
