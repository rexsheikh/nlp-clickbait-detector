# Naive Bayes
import sys
import argparse
import random
import re
from pathlib import Path

import nltk
# nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# set repo root for dataloaders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import dataset loaders and paths
from utility.dataLoader import load_texts_labels as load_texts_labels_unified  # noqa: E402

# ------------------------
# Tokenization and vocab
# ------------------------

TOP_N_DEFAULT = 2000

def regex_tokenize(text):
    # Lowercase alphanumerics and apostrophes
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def get_stopwords_set(enable=True):
    if not enable:
        return set()
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception as e:
        print(f"[warn] NLTK stopwords not available ({e}); continuing without stopword removal.")
        return set()


def build_word_features(texts, top_n=TOP_N_DEFAULT, stopwords_set=None):
    """
    Build top-N vocab using nltk.FreqDist over tokens.
    If stopwords_set is provided, tokens present in it are skipped.
    """
    stopwords_set = stopwords_set or set()
    # Accumulate tokens across all documents
    all_tokens = []
    for t in texts:
        toks = regex_tokenize(t)
        if stopwords_set:
            toks = [w for w in toks if w not in stopwords_set]
        all_tokens.extend(toks)
    # Use FreqDist like Problem4
    fd = nltk.FreqDist(all_tokens)
    word_features = list(fd)[:top_n]
    return word_features


def make_document_features_fn(word_features, stopwords_set=None):
    """
    Return a closure that maps tokens to a boolean feature dict contains(word),
    optionally removing stopwords prior to containment checks.
    """
    stopwords_set = stopwords_set or set()
    wf_set = set(word_features)

    def document_features_from_text(text):
        toks = regex_tokenize(text)
        if stopwords_set:
            toks = [w for w in toks if w not in stopwords_set]
        document_words = set(toks)
        # Boolean presence features for words in the selected word_features
        features = {}
        for w in wf_set:
            features[f"contains({w})"] = (w in document_words)
        return features

    return document_features_from_text


# ------------------------
# Data loading and splits
# ------------------------

def get_texts_labels_for(dataset):
    return load_texts_labels_unified(dataset)


def stratified_holdout(X, y, test_size=0.2, random_state=42):
    try:
        from sklearn.model_selection import train_test_split as sk_split
        return sk_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception:
        # Fallback: manual stratified split
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


def contiguous_holdout(X, y, frac=0.2):
    n = len(X)
    k = max(1, int(n * (1 - frac)))
    return X[:k], X[k:], y[:k], y[k:]


def choose_split(dataset, X, y):
    """
    Match conventions:
      - Kaggle/Webis: stratified shuffled 80/20 split
      - Train2: deterministic contiguous 80/20 split
    """
    ds = dataset.lower()
    if ds == "train2":
        return contiguous_holdout(X, y, frac=0.2)
    else:
        return stratified_holdout(X, y, test_size=0.2, random_state=42)


# ------------------------
# Training and evaluation
# ------------------------

def evaluate(gold, pred, tag):
    acc = accuracy_score(gold, pred)
    prec = precision_score(gold, pred, pos_label=1)
    rec = recall_score(gold, pred, pos_label=1)
    cm = confusion_matrix(gold, pred, labels=[0, 1])
    print(f"[{tag}] Acc={acc:.3f}  Prec(pos=1)={prec:.3f}  Rec(pos=1)={rec:.3f}")
    print(f"[{tag}] Confusion Matrix:\n{cm}")


def train_and_evaluate_naive_bayes(dataset, X_texts, y, top_n=TOP_N_DEFAULT, use_stopwords=False):
    """
    Build Problem4-style features, train NLTK NaiveBayes, and print metrics.
    When use_stopwords=False: baseline (no stopword removal).
    When use_stopwords=True: remove stopwords in vocab building and featurization.
    """
    X_train, X_test, y_train, y_test = choose_split(dataset, X_texts, y)

    stopwords_set = get_stopwords_set(enable=use_stopwords)
    word_features = build_word_features(X_train, top_n=top_n, stopwords_set=stopwords_set)
    feat_fn = make_document_features_fn(word_features, stopwords_set=stopwords_set)

    train_set = [(feat_fn(t), yv) for t, yv in zip(X_train, y_train)]
    test_feats = [feat_fn(t) for t in X_test]

    nb = NaiveBayesClassifier.train(train_set)
    pred = [nb.classify(x) for x in test_feats]

    mode = "StopwordsRemoved" if use_stopwords else "Baseline"
    tag = f"{dataset}][NaiveBayes][{mode}][topN={len(word_features)}"
    evaluate(y_test, pred, tag)


def run_for_dataset(dataset, top_n):
    print(f"\n=== {dataset.capitalize()} (Naive Bayes) ===")
    X_texts, y = get_texts_labels_for(dataset)
    if not X_texts or not y:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    # Baseline (no stopwords)
    train_and_evaluate_naive_bayes(dataset, X_texts, y, top_n=top_n, use_stopwords=False)
    # With stopword removal
    train_and_evaluate_naive_bayes(dataset, X_texts, y, top_n=top_n, use_stopwords=True)


def main():
    parser = argparse.ArgumentParser(description="Problem4-style Naive Bayes for clickbait datasets")
    parser.add_argument("--dataset", choices=["all", "kaggle", "train2", "webis"], default="all", help="Dataset to run")
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT, help="Top-N features for vocabulary (FreqDist)")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["kaggle", "train2", "webis"]:
            run_for_dataset(ds, args.top_n)
    else:
        run_for_dataset(args.dataset, args.top_n)


if __name__ == "__main__":
    main()
