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
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

# set repo root for dataloaders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# import dataset loaders and paths
from utility.dataLoader import load_texts_labels as load_texts_labels_unified

# == tokenization and vocab building ==
TOP_N_DEFAULT = 2000

TOP_IDENTS = 10

IGNORE_TERMS = {
    "the","a","an","this","that","these","those",
    "you","your","yours","what","which","who","whom",
    "and","or","but",
    "to","of","for","in","on","at","by","with","about","as","from",
    "is","are","was","were","be","been","being",
    "have","has","had","do","does","did",
    "it","its","they","them","their","theirs",
    "we","us","our","ours","i","me","my","mine",
    "he","him","his","she","her","hers",
    "yourself","yourselves","ourselves","himself","herself","themselves"
}

SUPERLATIVE_TERMS = {
    "best","top","most","greatest","ultimate","amazing","incredible","unbelievable","shocking",
    "craziest","wildest","epic","insane","must-see"
}
# description: gives word tokens from a given input text
# params: text (str)
# return: list[str]
def tokenize(text):
    return nltk.word_tokenize(text)

# description: return extended token stream with n-grams up to ngram_max
# params: tokens (list[str]), ngram_max (int)
# return: list[str]
def tokens_with_ngrams(tokens, ngram_max=1):
    tokens = list(tokens)
    if ngram_max is None or int(ngram_max) <= 1:
        return tokens
    toks = list(tokens)
    joined = list(tokens)
    for n in range(2, int(ngram_max) + 1):
        if len(toks) >= n:
            for i in range(len(toks) - n + 1):
                joined.append("_".join(toks[i:i+n]))
    return joined

# description: filter tokens by IGNORE_TERMS constant
# params: tokens (list[str])
# return: list[str]
def filter_ignored(tokens):
    return [w for w in tokens if w.lower() not in IGNORE_TERMS]

# description: checks input text against superlative terms constant set
# params: text (str), super_terms (set[str])
# return: bool
def contains_superlative(text, super_terms=SUPERLATIVE_TERMS):
    toks = tokenize(text or "")
    return any(t.lower() in super_terms for t in toks)

# description: gets stopwords if enabled is set to True
def get_stopwords(enabled):
    try:
        return set(stopwords.words("english")) if enabled else set()
    except Exception:
        return set()


# description: compute extra boolean features for punctuation/structure
# params: text (str), include_punct (bool), include_struct (bool)
# return: dict[str, bool|int|float]
def compute_extra_features(text, include_punct=False, include_struct=False):
    s = text or ""
    feats = {}
    if include_punct:
        ex_cnt = s.count("!")
        qm_cnt = s.count("?")
        feats["exclaim_present"] = ex_cnt > 0
        feats["qmark_present"] = qm_cnt > 0
    if include_struct:
        tokens = re.findall(r"[A-Za-z0-9']+", s)
        feats["digit_present"] = any(ch.isdigit() for ch in s)
        feats["superlative_present"] = contains_superlative(s)
        feats["short_length"] = (len(tokens) <= 6)
    return feats

# builds top-N vocab over tokens (+ optional n-grams)
def build_word_features(texts, top_n=TOP_N_DEFAULT, stopwords_set=None, ngram_max=1):
    stopwords_set = stopwords_set or set()
    all_tokens = []
    for t in texts:
        toks = tokenize(t)
        if stopwords_set:
            toks = [w for w in toks if w not in stopwords_set]
        toks = filter_ignored(toks)
        toks = tokens_with_ngrams(toks, ngram_max=ngram_max)
        all_tokens.extend(toks)
    fd = nltk.FreqDist(all_tokens)
    word_features = list(fd)[:top_n]
    return word_features

# description: create document->feature mapping function
# params: word_features (list[str]), stopwords_set (set), ngram_max (int), include_punct (bool), include_struct (bool)
#         ngram_max (int), include_punct (bool), include_struct (bool)
# return: dict feature_name -> bool/int/float
def build_document_features(text, word_features, stopwords_set=None, ngram_max=1, include_punct=False, include_struct=False):
    stopwords_set = stopwords_set or set()
    toks = tokenize(text)
    if stopwords_set:
        toks = [w for w in toks if w not in stopwords_set]
    toks = filter_ignored(toks)
    toks = tokens_with_ngrams(toks, ngram_max=int(ngram_max))
    document_words = set(toks)
    wf_set = set(word_features)
    features = {f"contains({w})": (w in document_words) for w in wf_set}
    if include_punct or include_struct:
        features.update(compute_extra_features(text, include_punct=include_punct, include_struct=include_struct))
    return features

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
        try:
            # Per-class metrics
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
        except Exception as e:
            print(f"[{tag}] Extended metrics error: {e}")

# description: Print top identifiers for Naive Bayes; uses NB's most_informative_features when available.
# params: nb (NaiveBayesClassifier), dataset_tag (str), top_k (int=TOP_IDENTS), word_features (list[str]|None)
# return: None (prints to stdout)
def show_top_identifiers(nb, dataset_tag, top_k=TOP_IDENTS, word_features=None):
    try:
        # NLTK prints directly; this will emit to stdout
        nb.show_most_informative_features(top_k)
    except Exception:
        # Fallback: show first top_k vocab entries if provided
        if word_features:
            print(f"[{dataset_tag}][NaiveBayes] Top {top_k} vocab features (by FreqDist order):")
            for w in word_features[:top_k]:
                print(f"  {w}")
        else:
            print(f"[{dataset_tag}][NaiveBayes] Unable to display top identifiers.")

# description: Train/evaluate NLTK Naive Bayes with Problem4-style boolean features on a fixed 80/20 split.
# params: dataset: e.g. kaggle, webis,
#         X_texts: list of raw text docs,
#         y: list of integer labels in X_texts, 
#         top_n: default 2000 vocab words
#         other params are boolean toggles for different experiments
# return: None (prints metrics to stdout)
def train_and_evaluate_naive_bayes(
    dataset,
    X_texts,
    y,
    top_n=TOP_N_DEFAULT,
    use_stopwords=False,
    ngram_max=1,
    punct_signals=False,
    struct_features=False,
    show_identifiers=False,
    top_k=10,
):
    # shuffle to ensure good train/test split results
    docs = list(zip(X_texts, y))
    rnd = random.Random(42)
    rnd.shuffle(docs)
    X_texts = []
    y = []
    for pair in docs:
        X_texts.append(pair[0])
        y.append(pair[1])

    n = len(X_texts)
    if n == 0:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    k = max(1, int(0.8 * n))

    # create vocab on training slice only then featurize
    stopwords_set = get_stopwords(use_stopwords)
    word_features = build_word_features(X_texts[:k], top_n=top_n, stopwords_set=stopwords_set, ngram_max=ngram_max)
    featuresets = [
        (
            build_document_features(
                t,
                word_features,
                stopwords_set=stopwords_set,
                ngram_max=ngram_max,
                include_punct=punct_signals,
                include_struct=struct_features,
            ),
            yv,
        )
        for t, yv in zip(X_texts, y)
    ]
    train_set, test_set = featuresets[:k], featuresets[k:]

    # print test distribution
    y_test = [yv for (_, yv) in test_set]
    pos_test = sum(1 for v in y_test if v == 1)
    neg_test = len(y_test) - pos_test
    print(f"[{dataset}] Test split distribution: pos={pos_test} neg={neg_test} (n={len(y_test)})")

    nb = NaiveBayesClassifier.train(train_set)
    gold = y_test
    pred = [nb.classify(x) for (x, _) in test_set]

    # scores for AUC
    try:
        y_score = [nb.prob_classify(x).prob(1) for (x, _) in test_set]
    except Exception:
        y_score = None

    mode = "StopwordsRemoved" if use_stopwords else "Baseline"
    tag = f"{dataset}][NaiveBayes][{mode}][topN={len(word_features)}][ng(1,{ngram_max})]"
    evaluate(gold, pred, tag, y_score=y_score)

    if show_identifiers:
        show_top_identifiers(nb, dataset.capitalize(), top_k=top_k, word_features=word_features)

# description: execute naive bayes as baseline. CLI args allow for configurable tuning
def run_for_dataset(dataset, args):
    print(f"\n=== {dataset.capitalize()} (Naive Bayes) ===")
    X_texts, y = get_texts_labels_for(dataset)
    if not X_texts or not y:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    train_and_evaluate_naive_bayes(
        dataset,
        X_texts,
        y,
        top_n=args.top_n,
        use_stopwords=getattr(args, "use_stopwords", False),
        ngram_max=args.ngram_max,
        punct_signals=args.punct_signals,
        struct_features=args.struct_features,
        show_identifiers=args.show_identifiers,
        top_k=args.top_k,
    )

def main():
    parser = argparse.ArgumentParser(description="Problem4-style Naive Bayes for clickbait datasets")
    parser.add_argument("--dataset", choices=["all", "kaggle", "train2", "webis"], default="all", help="Dataset to run")
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT, help="Top-N features for vocabulary (FreqDist)")
    parser.add_argument("--ngram-max", type=int, default=1, help="Max n for word n-grams in NB token stream (1..n)")
    # Factor toggles (subset applicable to NB)
    parser.add_argument("--punct-signals", action="store_true", help="Add punctuation signals (exclaim/qmark) as boolean features")
    parser.add_argument("--struct-features", action="store_true", help="Add structure features (digit/superlative/short_length) as boolean features")
    parser.add_argument("--use-stopwords", action="store_true", help="Use NLTK stopword removal during vocab/featurization")
    # Diagnostics
    parser.add_argument("--show-identifiers", action="store_true", help="Show top-N informative NB features (most_informative_features)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K terms to display in identifier printouts")

    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["kaggle", "train2", "webis"]:
            run_for_dataset(ds, args)
    else:
        run_for_dataset(args.dataset, args)

if __name__ == "__main__":
    main()
