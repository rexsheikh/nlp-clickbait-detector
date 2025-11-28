#!/usr/bin/env python3
# Linear SVM (SGD) runner for clickbait datasets
# - Uses TF-IDF features + SGDClassifier(loss="hinge") i.e., linear SVM with stochastic gradient descent
# - Runs on Kaggle, Train2, and Webis datasets
# - Mirrors structure/CLI of logisticRegression.py for easy parity
# - Keeps all imports at top

import sys
import argparse
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline as mkpipe
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Make repo root importable so we can import dataset loaders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Unified data loader
from utility.dataLoader import load_texts_labels as load_texts_labels_unified  # noqa: E402


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
# SVM (SGD) pipeline and metrics
# ------------------------

def build_svm_pipeline(ngram_max=2, min_df=2, max_df=1.0, alpha=1e-4, penalty="l2", l1_ratio=0.15, class_weight=None):
    # Vectorizer params
    vec_kwargs = {
        "ngram_range": (1, int(ngram_max)),
        "min_df": int(min_df),
    }
    if max_df is not None and float(max_df) != 1.0:
        vec_kwargs["max_df"] = float(max_df)

    # Class weight
    cw = None if (class_weight is None or str(class_weight).lower() == "none") else "balanced"

    clf = SGDClassifier(
        loss="hinge",          # linear SVM objective
        alpha=float(alpha),    # L2 regularization strength
        penalty=str(penalty),
        l1_ratio=float(l1_ratio),
        max_iter=1000,
        tol=1e-3,
        class_weight=cw,
        random_state=42,
    )

    pipe = mkpipe(TfidfVectorizer(**vec_kwargs), clf)
    return pipe


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
            from sklearn.metrics import (
                precision_recall_fscore_support,
                roc_auc_score,
                average_precision_score,
            )
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


# ------------------------
# Feature introspection (works for linear models with coef_)
# ------------------------

def get_vec_clf_linear(pipeline):
    """
    Extract TfidfVectorizer and linear classifier (SGDClassifier/LogisticRegression) from a sklearn Pipeline.
    """
    vec = getattr(pipeline, "named_steps", {}).get("tfidfvectorizer")
    clf = None
    # Try common names
    clf = getattr(pipeline, "named_steps", {}).get("sgdclassifier", None) or getattr(pipeline, "named_steps", {}).get("logisticregression", None)
    if vec is None or clf is None:
        for _, step in getattr(pipeline, "named_steps", {}).items():
            n = step.__class__.__name__.lower()
            if n == "tfidfvectorizer":
                vec = step
            if n in {"sgdclassifier", "logisticregression"}:
                clf = step
    return vec, clf


def print_top_linear_identifiers(pipeline, dataset_tag, top_k=10, mode="all"):
    """
    Print top positive/negative weighted features from a linear model with coef_.
    mode in {"all", "filtered", "bigrams"}
    """
    try:
        vec, clf = get_vec_clf_linear(pipeline)
        if vec is None or clf is None or not hasattr(clf, "coef_"):
            print(f"[{dataset_tag}][SVM-SGD][{mode}] Unable to extract feature names or coefficients.")
            return

        feature_names = vec.get_feature_names_out()
        coefs = clf.coef_[0]

        if mode == "bigrams":
            kept_indices = [i for i, name in enumerate(feature_names) if " " in name]
        elif mode == "filtered":
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
            kept_indices = [i for i, name in enumerate(feature_names) if name.lower() not in IGNORE_TERMS]
        else:
            kept_indices = list(range(len(feature_names)))

        if not kept_indices:
            print(f"[{dataset_tag}][SVM-SGD][{mode}] No terms left after filtering.")
            return

        pos_sorted = sorted(kept_indices, key=lambda i: coefs[i], reverse=True)
        neg_sorted = sorted(kept_indices, key=lambda i: coefs[i])

        print(f"[{dataset_tag}][SVM-SGD][{mode}] Top {top_k} clickbait indicators:")
        for i in pos_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")

        print(f"[{dataset_tag}][SVM-SGD][{mode}] Top {top_k} news indicators:")
        for i in neg_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")
    except Exception as e:
        print(f"[{dataset_tag}][SVM-SGD][{mode}] Error printing top identifiers: {e}")


# ------------------------
# Orchestration
# ------------------------

def train_and_evaluate_svm(dataset, X_texts, y, args):
    """
    Build TF-IDF + SGDClassifier(hinge) pipeline, train on train split, evaluate on holdout.
    Fixed 80/20 contiguous split by slicing.
    Optionally print top identifiers.
    """
    n = len(X_texts)
    if n == 0:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    k = max(1, int(0.8 * n))
    X_train, X_test = X_texts[:k], X_texts[k:]
    y_train, y_test = y[:k], y[k:]

    # Log test distribution
    pos_test = sum(1 for v in y_test if v == 1)
    neg_test = len(y_test) - pos_test
    print(f"[{dataset}] Test split distribution: pos={pos_test} neg={neg_test} (n={len(y_test)})")

    pipe = build_svm_pipeline(
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        alpha=args.alpha,
        penalty=args.penalty,
        l1_ratio=args.l1_ratio,
        class_weight=args.class_weight,
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # Continuous scores for AUCs (decision function margin works for ROC/PR AUC)
    try:
        y_score = pipe.decision_function(X_test)
    except Exception:
        y_score = None

    tag = f"{dataset}][SVM-SGD][ng(1,{args.ngram_max})_minDF{args.min_df}_alpha{args.alpha}_{args.penalty}"
    evaluate(y_test, preds, tag, y_score=y_score)

    if args.show_identifiers:
        mode = args.show_identifiers_mode
        top_k = args.top_k
        print_top_linear_identifiers(pipe, dataset.capitalize(), top_k=top_k, mode=mode)


def run_for_dataset(dataset, args):
    print(f"\n=== {dataset.capitalize()} (SVM-SGD) ===")
    X_texts, y = get_texts_labels_for(dataset)
    if not X_texts or not y:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    train_and_evaluate_svm(dataset, X_texts, y, args)


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + Linear SVM (SGD) for clickbait datasets")
    parser.add_argument("--dataset", choices=["all", "kaggle", "train2", "webis"], default="all", help="Dataset to run")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n for TF-IDF n-gram range (1..n)")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF")
    parser.add_argument("--max-df", type=float, default=1.0, help="Max document frequency for TF-IDF (<=1.0 means proportion)")

    # SGD (SVM) hyperparameters
    parser.add_argument("--alpha", type=float, default=1e-4, help="Regularization strength (SGD alpha)")
    parser.add_argument("--penalty", choices=["l2", "l1", "elasticnet"], default="l2", help="SGD penalty")
    parser.add_argument("--l1-ratio", type=float, default=0.15, help="L1 ratio (only used with elasticnet)")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Class weight setting")

    # Diagnostics
    parser.add_argument("--show-identifiers", action="store_true", help="Print top positive/negative weighted features")
    parser.add_argument("--show-identifiers-mode", choices=["all", "filtered", "bigrams"], default="all", help="Identifier filtering mode")
    parser.add_argument("--top-k", type=int, default=10, help="Top K terms to display in identifier printouts")

    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["kaggle", "train2", "webis"]:
            run_for_dataset(ds, args)
    else:
        run_for_dataset(args.dataset, args)


if __name__ == "__main__":
    main()
