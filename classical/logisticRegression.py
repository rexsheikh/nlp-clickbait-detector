# Logistic Regression runner (TF-IDF + LR) for clickbait datasets
# - Runs on Kaggle, Train2, and Webis datasets
# - Adapts patterns from classical_nlp_kaggle/webis and hw4/LR_Homework4.py
# - Provides CLI for vectorizer/LR hyperparameters and optional feature introspection
# - Keeps all imports at top

import sys
import argparse
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline as mkpipe
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Unified data loader
from utility.dataLoader import load_texts_labels as load_texts_labels_unified


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
# LR pipeline and metrics
# ------------------------

def build_lr_pipeline(ngram_max=2, min_df=2, max_df=1.0, C=1.0, class_weight=None):
    # Vectorizer params
    vec_kwargs = {
        "ngram_range": (1, int(ngram_max)),
        "min_df": int(min_df),
    }
    # Apply max_df only if < 1.0 or > 1 (to avoid no-op noise)
    if max_df is not None and float(max_df) != 1.0:
        vec_kwargs["max_df"] = float(max_df)

    # LR params
    cw = None if (class_weight is None or str(class_weight).lower() == "none") else "balanced"
    clf = LogisticRegression(max_iter=1000, C=float(C), class_weight=cw)

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
# Feature introspection
# ------------------------

def get_vec_clf(pipeline):
    """
    Extract TfidfVectorizer and LogisticRegression from a sklearn Pipeline; tolerant of step names.
    """
    vec = getattr(pipeline, "named_steps", {}).get("tfidfvectorizer")
    clf = getattr(pipeline, "named_steps", {}).get("logisticregression")
    if vec is None or clf is None:
        for _, step in getattr(pipeline, "named_steps", {}).items():
            n = step.__class__.__name__.lower()
            if n == "tfidfvectorizer":
                vec = step
            if n == "logisticregression":
                clf = step
    return vec, clf


def print_top_lr_identifiers(pipeline, dataset_tag, top_k=10, mode="all"):
    """
    Print top positive/negative weighted features from LogisticRegression.
    mode in {"all", "filtered", "bigrams"}
    """
    try:
        vec, clf = get_vec_clf(pipeline)
        if vec is None or clf is None:
            print(f"[{dataset_tag}][LogReg][{mode}] Unable to extract feature names or coefficients.")
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
            print(f"[{dataset_tag}][LogReg][{mode}] No terms left after filtering.")
            return

        pos_sorted = sorted(kept_indices, key=lambda i: coefs[i], reverse=True)
        neg_sorted = sorted(kept_indices, key=lambda i: coefs[i])

        print(f"[{dataset_tag}][LogReg][{mode}] Top {top_k} clickbait indicators:")
        for i in pos_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")

        print(f"[{dataset_tag}][LogReg][{mode}] Top {top_k} news indicators:")
        for i in neg_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")
    except Exception as e:
        print(f"[{dataset_tag}][LogReg][{mode}] Error printing top identifiers: {e}")


# ------------------------
# Orchestration
# ------------------------

# description: Train/evaluate TF-IDF + Logistic Regression on a deterministic 80/20 split with extended metrics.
# params: dataset (str), X_texts (list[str]), y (list[int]), args (Namespace with vectorizer/LR params)
# return: None (prints metrics to stdout)
def train_and_evaluate_lr(dataset, X_texts, y, args):
    # Shuffle deterministically to avoid ordered-split pathologies
    docs = list(zip(X_texts, y))
    rnd = random.Random(42)
    rnd.shuffle(docs)
    X_texts, y = zip(*docs) if docs else ([], [])

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

    pipe = build_lr_pipeline(
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        C=args.C,
        class_weight=args.class_weight,
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # Continuous scores for AUCs
    try:
        y_score = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = None

    tag = f"{dataset}][LogReg][ng(1,{args.ngram_max})_minDF{args.min_df}_C{args.C}"
    evaluate(y_test, preds, tag, y_score=y_score)

    if args.show_identifiers:
        mode = args.show_identifiers_mode
        top_k = args.top_k
        print_top_lr_identifiers(pipe, dataset.capitalize(), top_k=top_k, mode=mode)


def run_for_dataset(dataset, args):
    print(f"\n=== {dataset.capitalize()} (Logistic Regression) ===")
    X_texts, y = get_texts_labels_for(dataset)
    if not X_texts or not y:
        print(f"[{dataset}] No data loaded; skipping.")
        return
    train_and_evaluate_lr(dataset, X_texts, y, args)


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression for clickbait datasets")
    parser.add_argument("--dataset", choices=["all", "kaggle", "train2", "webis"], default="all", help="Dataset to run")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n for TF-IDF n-gram range (1..n)")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF")
    parser.add_argument("--max-df", type=float, default=1.0, help="Max document frequency for TF-IDF (<=1.0 means proportion)")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Class weight setting")
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
