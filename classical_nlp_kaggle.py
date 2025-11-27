import random
import csv
import re
import json
import datetime
from collections import Counter
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# === Data Setup ===
# relative to project root, ensure that data is placed in data directory
# reference readme for data download instructions
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

KAGGLE_DATASET_PATH = str(DATA_DIR / "kaggle_clickbait.csv")
TRAIN2_DATASET_PATH = str(DATA_DIR / "news_clickbait_dataset" / "train2.csv")


# === Load Data ===
# loader returning (texts, labels).
# params: path (string), schema ("kaggle" or "train2")
# return: texts (list of str), labels (list of int where 1=clickbait, 0=news)
def load_texts_labels(path, schema):
    texts = []
    labels = []
    if schema == "kaggle":
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("clickbait")
                text = row.get("headline")
                if raw is None or text is None:
                    continue
                try:
                    label = int(str(raw).strip().lower())
                except Exception:
                    continue
                texts.append(str(text).strip())
                labels.append(label)
    elif schema == "train2":
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("label")
                text = row.get("title")
                if raw is None or text is None:
                    continue
                label = str(raw).strip().lower()
                label = 1 if label == "clickbait" else 0
                texts.append(str(text).strip())
                labels.append(label)
    else:
        raise ValueError(f"Unknown schema: {schema}")
    return texts, labels


# sum class counts from labels produced by load_texts_labels.
# params: path (string), schema ("kaggle" or "train2")
# return dict (clickbait: clickbait count, news: news_count)
def load_counts(path, schema):
    _, labels = load_texts_labels(path, schema)
    ctr = Counter(labels)
    return {"clickbait": ctr.get(1,0), "news": ctr.get(0,0)}


# tokenization
TOP_N_FEATURES = 2000


# word-level tokenization
# Input: text (string)
# Output: list of tokens (list of str)
def regex_tokenize(text):
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


# gets top-N vocabulary from texts using regex_tokenize
# params: texts (list of str), top_n (int)
# return: vocab (list of str)
def build_vocab(texts, top_n=TOP_N_FEATURES):
    cnt = Counter()
    for t in texts:
        cnt.update(regex_tokenize(t))
    return [w for w, _ in cnt.most_common(top_n)]


# Map vocabulary presence into boolean feature dict.
# Input: tokens (list of str), vocab (list of str)
# Output: dict feature_name -> bool
def document_features(tokens, vocab):
    token_set = set(tokens)
    return {f"document features contains({w})": (w in token_set) for w in vocab}


# =====================================================================
# Pipeline helpers and diagnostics
# =====================================================================

# Extract TfidfVectorizer and LogisticRegression from a sklearn Pipeline; tolerant of step names.
# Input: pipeline (sklearn Pipeline)
# Output: tuple (vectorizer, classifier) or (None, None)
def get_vec_clf(pipeline):
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


# Print top positive/negative weighted features from a LogisticRegression pipeline.
# Input: pipeline, dataset_tag (str), top_k (int), mode ("all"|"filtered"|"bigrams")
# Output: prints to stdout
def _print_top_lr_identifiers_core(pipeline, dataset_tag, top_k, mode):
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


# Backwards-compatible wrappers (keep existing names)
def print_top_lr_identifiers(pipeline, dataset_tag, top_k=10):
    _print_top_lr_identifiers_core(pipeline, dataset_tag, top_k, mode="all")


def print_top_lr_identifiers_filtered(pipeline, dataset_tag, top_k=10):
    _print_top_lr_identifiers_core(pipeline, dataset_tag, top_k, mode="filtered")


def print_top_lr_identifiers_bigrams_only(pipeline, dataset_tag, top_k=10):
    _print_top_lr_identifiers_core(pipeline, dataset_tag, top_k, mode="bigrams")


# =====================================================================
# Metrics Writer (JSONL)
# =====================================================================

# Append a JSON record for either a holdout evaluation (y_true/y_pred provided)
# or a CV-best summary (best_f1/params provided).
# Input: dataset (str), split ("holdout"|"cv-best"), model (str), variant (str),
#        y_true/y_pred for holdout OR best_f1/params for CV; path to JSONL file.
# Output: appends one JSON line to the metrics file.
def write_metrics_record(
    dataset,
    split,
    model,
    variant,
    y_true=None,
    y_pred=None,
    best_f1=None,
    params=None,
    path="metrics.jsonl",
):
    record = {
        "dataset": dataset,
        "split": split,  # "holdout" or "cv-best"
        "model": model,  # "NLTK_NB" | "MNB" | "LogReg"
        "variant": variant,
        "metrics": {},
        "confusion": None,
        "params": params or {},
        "run_id": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if y_true is not None and y_pred is not None:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1)
        rec = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        from sklearn.metrics import confusion_matrix as _cm
        record["metrics"] = {
            "accuracy": float(acc),
            "precision_pos1": float(prec),
            "recall_pos1": float(rec),
            "f1_pos1": float(f1),
        }
        cm = _cm(y_true, y_pred, labels=[0, 1]).tolist()
        record["confusion"] = cm
    elif best_f1 is not None:
        record["metrics"] = {
            "accuracy": None,
            "precision_pos1": None,
            "recall_pos1": None,
            "f1_pos1": float(best_f1),
        }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Standardized evaluation printing + JSONL recording for holdout predictions.
# Input: dataset_tag (str), model_name (str), variant (str), y_true, y_pred, path (str)
# Output: prints metrics and writes one JSONL record.
def evaluate_and_record(dataset_tag, model_name, variant, y_true, y_pred, path="metrics.jsonl"):
    preds = y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred
    print(
        f"[{dataset_tag}][{model_name}]   Acc={accuracy_score(y_true, preds):.3f}  "
        f"Prec(pos)={precision_score(y_true, preds, pos_label=1):.3f}  "
        f"Rec(pos)={recall_score(y_true, preds, pos_label=1):.3f}"
    )
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print(f"[{dataset_tag}][{model_name}] Confusion Matrix:\n{cm}")
    write_metrics_record(
        dataset=dataset_tag,
        split="holdout",
        model=model_name,
        variant=variant,
        y_true=y_true,
        y_pred=preds,
        path=path,
    )


# =====================================================================
# Train/test split helpers
# =====================================================================

def _stratified_split_fallback(X, y, test_size=0.2, random_state=42):
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


def stratified_holdout(X, y, test_size=0.2, random_state=42):
    try:
        from sklearn.model_selection import train_test_split as sk_split
        return sk_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception:
        return _stratified_split_fallback(X, y, test_size=test_size, random_state=random_state)


def contiguous_holdout(X, y, frac=0.2):
    n = len(X)
    k = max(1, int(n * (1 - frac)))
    return X[:k], X[k:], y[:k], y[k:]


# =====================================================================
# Experiment runners
# =====================================================================

# Run baseline experiments (NLTK NB, MNB, LogReg) on holdout split.
# Input: X_texts (list of str), y (list of int), dataset_tag (str), split ("stratified" or "contiguous")
# Output: prints metrics and writes JSONL records.
def run_experiments(X_texts, y, dataset_tag, split):
    if not X_texts or not y:
        print(f"No data loaded from {dataset_tag} file; skipping experiments.")
        return

    if split == "stratified":
        X_train, X_test, y_train, y_test = stratified_holdout(X_texts, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = contiguous_holdout(X_texts, y, frac=0.2)

    # Boolean presence features over a top-N vocab
    vocab = build_vocab(X_train)

    def featurize(texts):
        return [document_features(regex_tokenize(t), vocab) for t in texts]

    # NLTK Naive Bayes
    try:
        from nltk.classify import NaiveBayesClassifier as NLTKNaiveBayes
        train_set = list(zip(featurize(X_train), y_train))
        test_feats = featurize(X_test)
        nb = NLTKNaiveBayes.train(train_set)
        pred_nb = [nb.classify(x) for x in test_feats]
        evaluate_and_record(dataset_tag, "NLTK_NB", "baseline_topN", y_test, pred_nb)
    except Exception as e:
        print(f"[{dataset_tag}][NLTK NaiveBayes] Error: {e}")

    # sklearn MultinomialNB with CountVectorizer baseline
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline
        mnb_pipeline = make_pipeline(
            CountVectorizer(max_features=TOP_N_FEATURES),
            MultinomialNB()
        )
        mnb_pipeline.fit(X_train, y_train)
        pred_mnb = mnb_pipeline.predict(X_test)
        evaluate_and_record(dataset_tag, "MNB", "count_maxfeat_topN", y_test, pred_mnb)
    except Exception as e:
        print(f"[{dataset_tag}][MultinomialNB] Error: {e}")

    # Logistic Regression with TF-IDF (n-grams)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline as mkpipe

        lr_pipeline = mkpipe(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=1000)
        )
        lr_pipeline.fit(X_train, y_train)
        pred_lr = lr_pipeline.predict(X_test)
        evaluate_and_record(dataset_tag, "LogReg", "tfidf_ng12_min2", y_test, pred_lr)

        # Diagnostics
        print_top_lr_identifiers(lr_pipeline, dataset_tag, top_k=10)
        print_top_lr_identifiers_filtered(lr_pipeline, dataset_tag, top_k=10)
        print_top_lr_identifiers_bigrams_only(lr_pipeline, dataset_tag, top_k=10)
    except Exception as e:
        print(f"[{dataset_tag}][LogReg] Error: {e}")


# Run Stratified K-Fold CV grid searches for LR and MNB.
# Input: X_texts (list of str), y (list of int), dataset_tag (str), shuffle (bool)
# Output: prints best scores/params and writes JSONL records.
def run_cv_grid(X_texts, y, dataset_tag, shuffle):
    print(f"\n=== {dataset_tag} CV + Grid (LR/MNB) ===")
    try:
        from sklearn.model_selection import StratifiedKFold, GridSearchCV

        # Logistic Regression TF-IDF pipeline/grid
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline as mkpipe
            lr_pipe = mkpipe(
                TfidfVectorizer(),
                LogisticRegression(max_iter=1000)
            )
            lr_grid = {
                "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
                "tfidfvectorizer__min_df": [1, 2, 5],
                "tfidfvectorizer__max_df": [0.9, 0.95],
                "logisticregression__C": [0.5, 1.0, 2.0],
                "logisticregression__class_weight": [None, "balanced"],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=42 if shuffle else None)
            lr_search = GridSearchCV(lr_pipe, lr_grid, cv=cv, scoring="f1", n_jobs=None)
            lr_search.fit(X_texts, y)
            print(f"[{dataset_tag}][LogReg][CV] best_f1={lr_search.best_score_:.3f}")
            print(f"[{dataset_tag}][LogReg][CV] best_params={lr_search.best_params_}")
            write_metrics_record(
                dataset=dataset_tag,
                split="cv-best",
                model="LogReg",
                variant="grid",
                best_f1=lr_search.best_score_,
                params=lr_search.best_params_,
            )
            best_lr = lr_search.best_estimator_
            print_top_lr_identifiers(best_lr, dataset_tag, top_k=10)
            print_top_lr_identifiers_filtered(best_lr, dataset_tag, top_k=10)
            print_top_lr_identifiers_bigrams_only(best_lr, dataset_tag, top_k=10)
        except Exception as e:
            print(f"[{dataset_tag}][LogReg][CV] Error: {e}")

        # MultinomialNB CountVectorizer pipeline/grid
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import make_pipeline
            mnb_pipe = make_pipeline(
                CountVectorizer(),
                MultinomialNB()
            )
            mnb_grid = {
                "countvectorizer__ngram_range": [(1, 1), (1, 2)],
                "countvectorizer__min_df": [1, 2, 5],
                "multinomialnb__alpha": [0.1, 0.5, 1.0, 2.0],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=42 if shuffle else None)
            mnb_search = GridSearchCV(mnb_pipe, mnb_grid, cv=cv, scoring="f1", n_jobs=None)
            mnb_search.fit(X_texts, y)
            print(f"[{dataset_tag}][MNB][CV]   best_f1={mnb_search.best_score_:.3f}")
            print(f"[{dataset_tag}][MNB][CV]   best_params={mnb_search.best_params_}")
            write_metrics_record(
                dataset=dataset_tag,
                split="cv-best",
                model="MNB",
                variant="grid",
                best_f1=mnb_search.best_score_,
                params=mnb_search.best_params_,
            )
        except Exception as e:
            print(f"[{dataset_tag}][MNB][CV] Error: {e}")
    except Exception as e:
        print(f"[{dataset_tag}][CV+Grid] Error: {e}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    # Kaggle dataset
    k_c1, k_c0 = load_counts(KAGGLE_DATASET_PATH, "kaggle")
    print(f"[Kaggle] counts - clickbait (label=1): {k_c1}")
    print(f"[Kaggle] counts - news      (label=0): {k_c0}")
    kaggle_texts, kaggle_y = load_texts_labels(KAGGLE_DATASET_PATH, "kaggle")
    print("\n=== Kaggle Clickbait Dataset ===")
    run_experiments(kaggle_texts, kaggle_y, dataset_tag="Kaggle", split="stratified")
    run_cv_grid(kaggle_texts, kaggle_y, dataset_tag="Kaggle", shuffle=True)

    # Train2 dataset (no shuffle split, contiguous)
    t_c1, t_c0 = load_counts(TRAIN2_DATASET_PATH, "train2")
    print(f"[Train2] counts - clickbait (label=1): {t_c1}")
    print(f"[Train2] counts - news      (label=0): {t_c0}")
    train2_texts, train2_y = load_texts_labels(TRAIN2_DATASET_PATH, "train2")
    print("\n=== Train2 Dataset ===")
    run_experiments(train2_texts, train2_y, dataset_tag="Train2", split="contiguous")
    run_cv_grid(train2_texts, train2_y, dataset_tag="Train2", shuffle=False)
