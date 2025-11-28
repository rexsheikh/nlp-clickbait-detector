# Naive Bayes
import sys
import argparse
import random
import re
from pathlib import Path

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

# set repo root for dataloaders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# import dataset loaders and paths
from utility.dataLoader import load_texts_labels as load_texts_labels_unified

# == tokenization and vocab building ==

TOP_N_DEFAULT = 2000

def tokenize(text):
    return nltk.word_tokenize(text)

# gets nltk's stopwords if enabled
def get_stopwords_set(enable=True):
    if not enable:
        return set()
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception as e:
        print(f"[warn] NLTK stopwords not available ({e}); continuing without stopword removal.")
        return set()

# builds top-N vocab over tokens
def build_word_features(texts, top_n=TOP_N_DEFAULT, stopwords_set=None):
    stopwords_set = stopwords_set or set()
    all_tokens = []
    for t in texts:
        toks = tokenize(t)
        if stopwords_set:
            toks = [w for w in toks if w not in stopwords_set]
        all_tokens.extend(toks)
    fd = nltk.FreqDist(all_tokens)
    word_features = list(fd)[:top_n]
    return word_features


def make_document_features_fn(word_features, stopwords_set=None):
    stopwords_set = stopwords_set or set()
    wf_set = set(word_features)

    def document_features_from_text(text):
        toks = tokenize(text)
        if stopwords_set:
            toks = [w for w in toks if w not in stopwords_set]
        document_words = set(toks)
        features = {}
        for w in wf_set:
            features[f"contains({w})"] = (w in document_words)
        return features

    return document_features_from_text

def get_texts_labels_for(dataset):
    return load_texts_labels_unified(dataset)

# == training and evaluation ==

def evaluate(gold, pred, tag, y_score=None):
    acc = accuracy_score(gold, pred)
    prec = precision_score(gold, pred, pos_label=1, zero_division=0)
    rec = recall_score(gold, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=[0, 1])
    print(f"[{tag}] Acc={acc:.3f}  Prec(pos=1)={prec:.3f}  Rec(pos=1)={rec:.3f}")
    print(f"[{tag}] Confusion Matrix:\n{cm}")

    # Extended metrics (only if a continuous score is provided)
    if y_score is not None:
        p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(
            gold, pred, labels=[0, 1], zero_division=0
        )
        # Macro/micro
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            gold, pred, average="macro", zero_division=0
        )
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            gold, pred, average="micro", zero_division=0
        )
        # AUCs (guard single-class)
        try:
            roc_auc = roc_auc_score(gold, y_score)
        except Exception:
            roc_auc = None
        try:
            pr_auc = average_precision_score(gold, y_score)
        except Exception:
            pr_auc = None

        print(f"[{tag}] Per-class:")
        print(f"  class 0: prec={p_c[0]:.3f} rec={r_c[0]:.3f} f1={f1_c[0]:.3f} support={supp_c[0]}")
        print(f"  class 1: prec={p_c[1]:.3f} rec={r_c[1]:.3f} f1={f1_c[1]:.3f} support={supp_c[1]}")
        print(f"[{tag}] Macro: prec={p_macro:.3f} rec={r_macro:.3f} f1={f1_macro:.3f}")
        print(f"[{tag}] Micro: prec={p_micro:.3f} rec={r_micro:.3f} f1={f1_micro:.3f}")
        print(f"[{tag}] ROC-AUC: {'N/A' if roc_auc is None else f'{roc_auc:.3f}'}")
        print(f"[{tag}] PR-AUC:  {'N/A' if pr_auc is None else f'{pr_auc:.3f}'}")



# description: Train/evaluate NLTK Naive Bayes with Problem4-style boolean features on a fixed 80/20 split.
# params: dataset (str), X_texts (list[str]), y (list[int]), top_n (int), use_stopwords (bool)
# return: None (prints metrics to stdout)
def train_and_evaluate_naive_bayes(dataset, X_texts, y, top_n=TOP_N_DEFAULT, use_stopwords=False):
    """
    Build Problem4-style features, train NLTK NaiveBayes, and print metrics.
    Deterministic shuffle + fixed 80/20 contiguous split by slicing.
    Vocab is built on the training slice only to avoid leakage, then applied to all texts before slicing features.
    """
    # Deterministically shuffle documents to avoid pathological ordered splits
    docs = list(zip(X_texts, y))
    rnd = random.Random(42)
    rnd.shuffle(docs)
    X_texts, y = zip(*docs) if docs else ([], [])

    n = len(X_texts)
    if n == 0:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    k = max(1, int(0.8 * n))

    # Build vocab on train slice only, then featurize all documents with that vocab
    stopwords_set = get_stopwords_set(enable=use_stopwords)
    word_features = build_word_features(X_texts[:k], top_n=top_n, stopwords_set=stopwords_set)
    feat_fn = make_document_features_fn(word_features, stopwords_set=stopwords_set)

    featuresets = [(feat_fn(t), yv) for t, yv in zip(X_texts, y)]
    train_set, test_set = featuresets[:k], featuresets[k:]

    # Log test distribution to confirm both classes present
    y_test = [yv for (_, yv) in test_set]
    pos_test = sum(1 for v in y_test if v == 1)
    neg_test = len(y_test) - pos_test
    print(f"[{dataset}] Test split distribution: pos={pos_test} neg={neg_test} (n={len(y_test)})")

    nb = NaiveBayesClassifier.train(train_set)
    gold = y_test
    pred = [nb.classify(x) for (x, _) in test_set]

    # Continuous scores for AUCs: P(class=1)
    try:
        y_score = [nb.prob_classify(x).prob(1) for (x, _) in test_set]
    except Exception:
        y_score = None

    mode = "StopwordsRemoved" if use_stopwords else "Baseline"
    tag = f"{dataset}][NaiveBayes][{mode}][topN={len(word_features)}"
    evaluate(gold, pred, tag, y_score=y_score)


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
