import sys
import argparse
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline as mkpipe, FeatureUnion, Pipeline as SKPipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.dataLoader import load_texts_labels as load_texts_labels_unified  # noqa: E402
from utility.common_text import compute_extra_features, TOP_IDENTS, IGNORE_TERMS





# ------------------------
# Data loading helper
# ------------------------

def get_texts_labels_for(dataset):
    return load_texts_labels_unified(dataset)


# ------------------------
# SVM (SGD) feature builders
# ------------------------

# description: Featurizer to extract structure/punctuation features from raw text.
# params: include_punct (bool), include_struct (bool)
# return: sklearn-compatible transformer producing list[dict]
class StructureFeaturizer:
    def __init__(self, include_punct=False, include_struct=False):
        self.include_punct = include_punct
        self.include_struct = include_struct

    def fit(self, X, y=None):
        return self


    def transform(self, X):
        out = []
        for t in X:
            s = t or ""
            d = compute_extra_features(s, include_punct=self.include_punct, include_struct=self.include_struct)
            out.append(d)
        return out


# description: Build a unioned feature extractor (word + optional char + optional structure) and SVM-SGD classifier.
# params: args (Namespace with flags), eff_min_df (int), eff_max_df (float|None)
# return: sklearn Pipeline
def build_svm_pipeline_from_args(args, eff_min_df, eff_max_df):
    branches = []
    # word branch
    stop_words = IGNORE_TERMS if getattr(args, "ignore_terms", False) else None
    word_vec = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=int(eff_min_df),
        stop_words=stop_words,
        **({} if eff_max_df is None else {"max_df": float(eff_max_df)})
    )
    branches.append(("word", word_vec))


    # structure branch
    include_punct = getattr(args, "punct_signals", False)
    include_struct = getattr(args, "struct_features", False)
    if include_punct or include_struct:
        struct_pipe = SKPipeline([
            ("fe", StructureFeaturizer(include_punct=include_punct, include_struct=include_struct)),
            ("dv", DictVectorizer())
        ])
        branches.append(("struct", struct_pipe))

    # union or single
    if len(branches) == 1:
        features = branches[0][1]
    else:
        features = FeatureUnion(branches)

    # classifier
    cw = None if (args.class_weight is None or str(args.class_weight).lower() == "none") else "balanced"
    clf = SGDClassifier(
        loss="hinge",
        alpha=float(args.alpha),
        penalty=str(args.penalty),
        l1_ratio=float(args.l1_ratio),
        max_iter=1000,
        tol=1e-3,
        class_weight=cw,
        random_state=42,
    )

    return SKPipeline([("features", features), ("clf", clf)])


# ------------------------
# Metrics
# ------------------------

# description: Print core and extended metrics; extended metrics only when y_score is provided.
# params: gold (list[int]), pred (list[int]), tag (str), y_score (array-like|None)
# return: None (prints to stdout)
def evaluate(gold, pred, tag, y_score=None):
    acc = accuracy_score(gold, pred)
    prec = precision_score(gold, pred, pos_label=1, zero_division=0)
    rec = recall_score(gold, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=[0, 1])
    print(f"[{tag}] Acc={acc:.3f}  Prec(pos=1)={prec:.3f}  Rec(pos=1)={rec:.3f}")
    print(f"[{tag}] Confusion Matrix:\n{cm}")

    if y_score is not None:
        try:
            p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(
                gold, pred, labels=[0, 1], zero_division=0
            )
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                gold, pred, average="macro", zero_division=0
            )
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                gold, pred, average="micro", zero_division=0
            )
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

# description: Print top positive/negative weighted word features for a linear model when a single TfidfVectorizer is used.
# params: pipeline (sklearn Pipeline), dataset_tag (str), top_k (int=TOP_IDENTS)
# return: None (prints to stdout)
def show_top_identifiers(pipeline, dataset_tag, top_k=TOP_IDENTS, ignore_terms=False):
    try:
        vec = getattr(pipeline, "named_steps", {}).get("features")
        clf = getattr(pipeline, "named_steps", {}).get("clf")

        # Only support a single TfidfVectorizer (word) branch for clear mapping
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        from sklearn.pipeline import FeatureUnion as _FU
        if isinstance(vec, _FU):
            print(f"[{dataset_tag}][SVM-SGD] Identifier printing not supported with multiple feature branches (FeatureUnion).")
            return
        if not isinstance(vec, _TV) or not hasattr(clf, "coef_"):
            print(f"[{dataset_tag}][SVM-SGD] Unable to extract word feature names or coefficients.")
            return

        feature_names = vec.get_feature_names_out()
        coefs = clf.coef_[0]

        if ignore_terms:
            kept_indices = [i for i, name in enumerate(feature_names) if name.lower() not in IGNORE_TERMS]
        else:
            kept_indices = list(range(len(feature_names)))
        if not kept_indices:
            print(f"[{dataset_tag}][SVM-SGD] No terms left after filtering.")
            return

        pos_sorted = sorted(kept_indices, key=lambda i: coefs[i], reverse=True)
        neg_sorted = sorted(kept_indices, key=lambda i: coefs[i])

        print(f"[{dataset_tag}][SVM-SGD] Top {top_k} clickbait indicators:")
        for i in pos_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")

        print(f"[{dataset_tag}][SVM-SGD] Top {top_k} news indicators:")
        for i in neg_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")
    except Exception as e:
        print(f"[{dataset_tag}][SVM-SGD] Error printing top identifiers: {e}")

# ------------------------
# Orchestration
# ------------------------

# description: Train/evaluate TF-IDF (+ optional char/struct) + Linear SVM (SGD) on a deterministic 80/20 split.
# params: dataset (str), X_texts (list[str]), y (list[int]), args (Namespace)
# return: None (prints metrics to stdout)
def train_and_evaluate_svm(dataset, X_texts, y, args):
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

    # Effective pruning settings
    eff_min_df = 5 if getattr(args, "min_df_high", False) else args.min_df
    eff_max_df = 0.95 if getattr(args, "min_df_high", False) else (None if args.max_df == 1.0 else args.max_df)

    pipe = build_svm_pipeline_from_args(args, eff_min_df=int(eff_min_df), eff_max_df=eff_max_df)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # Continuous scores for AUCs (decision function margin works for ROC/PR AUC)
    try:
        y_score = pipe.decision_function(X_test)
    except Exception:
        y_score = None

    tag = f"{dataset}][SVM-SGD][minDF{args.min_df}_alpha{args.alpha}_{args.penalty}"
    evaluate(y_test, preds, tag, y_score=y_score)

    if args.show_identifiers:
        top_k = args.top_k if hasattr(args, "top_k") else TOP_IDENTS
        show_top_identifiers(pipe, dataset.capitalize(), top_k=top_k, ignore_terms=getattr(args, "ignore_terms", False))


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
    parser.add_argument("--design", choices=["baseline","L8-1","L8-2","L8-3","L8-4","L8-5","L8-6","L8-7","L8-8"], default=None, help="Preset factor design")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for word TF-IDF")
    parser.add_argument("--max-df", type=float, default=1.0, help="Max document frequency for TF-IDF (<=1.0 means proportion)")

    # SGD (SVM) hyperparameters
    parser.add_argument("--alpha", type=float, default=1e-4, help="Regularization strength (SGD alpha)")
    parser.add_argument("--penalty", choices=["l2", "l1", "elasticnet"], default="l2", help="SGD penalty")
    parser.add_argument("--l1-ratio", type=float, default=0.15, help="L1 ratio (only used with elasticnet)")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Class weight setting")

    # Factor toggles
    parser.add_argument("--punct-signals", action="store_true", help="Add punctuation signals (exclaim/qmark/emoji)")
    parser.add_argument("--struct-features", action="store_true", help="Add length/structure features")
    parser.add_argument("--ignore-terms", action="store_true", default=False, help="Ignore common terms (IGNORE_TERMS) during featurization/identifiers")
    parser.add_argument("--min-df-high", action="store_true", help="Use higher pruning (min_df=5, max_df=0.95)")

    # Diagnostics
    parser.add_argument("--show-identifiers", action="store_true", help="Print top positive/negative weighted features (limited support under unions)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K terms to display in identifier printouts")

    args = parser.parse_args()

    # Apply design presets
    if args.design:
        presets = {
            "baseline": dict(punct_signals=False, struct_features=False, min_df_high=False, class_weight="none"),
            "L8-1": dict(punct_signals=False, struct_features=False, min_df_high=False, class_weight="none"),
            "L8-2": dict(punct_signals=False, struct_features=True,  min_df_high=True,  class_weight="none"),
            "L8-3": dict(punct_signals=True,  struct_features=True,  min_df_high=True,  class_weight="none"),
            "L8-4": dict(punct_signals=True,  struct_features=False, min_df_high=False, class_weight="none"),
            "L8-5": dict(punct_signals=False, struct_features=True,  min_df_high=True,  class_weight="balanced"),
            "L8-6": dict(punct_signals=False, struct_features=False, min_df_high=False, class_weight="balanced"),
            "L8-7": dict(punct_signals=True,  struct_features=False, min_df_high=False, class_weight="balanced"),
            "L8-8": dict(punct_signals=True,  struct_features=True,  min_df_high=True,  class_weight="balanced"),
        }
        cfg = presets.get(args.design, {})
        for k, v in cfg.items():
            setattr(args, k, v)

    if args.dataset == "all":
        for ds in ["kaggle", "train2", "webis"]:
            run_for_dataset(ds, args)
    else:
        run_for_dataset(args.dataset, args)


if __name__ == "__main__":
    main()
