import random
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import re
from collections import Counter

# constant path for kaggle dataset
KAGGLE_DATASET_PATH = "/Users/rexsheikh/Documents/boulder-fall-2025/nlp/nlp-clickbait-detector/data/kaggle_clickbait.csv"
TRAIN2_DATASET_PATH = "/Users/rexsheikh/Documents/boulder-fall-2025/nlp/nlp-clickbait-detector/data/news_clickbait_dataset/train2.csv"


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


# ================= Train2 loader and counters (label,text; no shuffle) =================

def load_data_train2(path: str = TRAIN2_DATASET_PATH):
    clickbait_1 = 0
    news_0 = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            label = (row[0] or "").lstrip("\ufeff").strip().lower()
            if label == "clickbait":
                clickbait_1 += 1
            elif label == "news":
                news_0 += 1
    return clickbait_1, news_0


def load_train2_texts_labels(path: str = TRAIN2_DATASET_PATH):
    texts, labels = [], []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            label = (row[0] or "").lstrip("\ufeff").strip().lower()
            if label not in {"clickbait", "news"}:
                continue
            text = ",".join(row[1:]).strip()
            y = 1 if label == "clickbait" else 0
            texts.append(text)
            labels.append(y)
    return texts, labels


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


def print_top_lr_identifiers(pipeline, dataset_tag: str, top_k: int = 10):
    """
    Print top K positive and negative weighted n-grams learned by LogisticRegression.
    Positive weights -> class 1 (clickbait); Negative weights -> class 0 (news).
    """
    try:
        vec = pipeline.named_steps.get("tfidfvectorizer")
        clf = pipeline.named_steps.get("logisticregression")
        if vec is None or clf is None:
            # Fallback: iterate steps to find by type name
            for name, step in pipeline.named_steps.items():
                if step.__class__.__name__.lower() == "tfidfvectorizer":
                    vec = step
                if step.__class__.__name__.lower() == "logisticregression":
                    clf = step
        if vec is None or clf is None:
            print(f"[{dataset_tag}][LogReg] Unable to extract feature names or coefficients.")
            return

        feature_names = vec.get_feature_names_out()
        coefs = clf.coef_[0]
        # Top positive (clickbait indicators)
        top_pos_idx = coefs.argsort()[-top_k:][::-1]
        # Top negative (news indicators)
        top_neg_idx = coefs.argsort()[:top_k]

        print(f"[{dataset_tag}][LogReg] Top {top_k} clickbait indicators:")
        for i in top_pos_idx:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")

        print(f"[{dataset_tag}][LogReg] Top {top_k} news indicators:")
        for i in top_neg_idx:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")
    except Exception as e:
        print(f"[{dataset_tag}][LogReg] Error printing top identifiers: {e}")


def print_top_lr_identifiers_filtered(pipeline, dataset_tag: str, top_k: int = 10):
    """
    Filter out very common function words/pronouns to surface more informative terms.
    Keeps numeric tokens and short tokens as requested.
    """
    try:
        vec = pipeline.named_steps.get("tfidfvectorizer")
        clf = pipeline.named_steps.get("logisticregression")
        if vec is None or clf is None:
            for _, step in pipeline.named_steps.items():
                if step.__class__.__name__.lower() == "tfidfvectorizer":
                    vec = step
                if step.__class__.__name__.lower() == "logisticregression":
                    clf = step
        if vec is None or clf is None:
            print(f"[{dataset_tag}][LogReg][Filtered] Unable to extract feature names or coefficients.")
            return

        feature_names = vec.get_feature_names_out()
        coefs = clf.coef_[0]

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

        # Keep features not in the ignore set. This only filters exact matches (mostly unigrams).
        kept_indices = [i for i, name in enumerate(feature_names) if name.lower() not in IGNORE_TERMS]
        if not kept_indices:
            print(f"[{dataset_tag}][LogReg][Filtered] No terms left after filtering.")
            return

        # Sort by weight among kept indices
        pos_sorted = sorted(kept_indices, key=lambda i: coefs[i], reverse=True)
        neg_sorted = sorted(kept_indices, key=lambda i: coefs[i])

        print(f"[{dataset_tag}][LogReg][Filtered] Top {top_k} clickbait indicators:")
        for i in pos_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")

        print(f"[{dataset_tag}][LogReg][Filtered] Top {top_k} news indicators:")
        for i in neg_sorted[:top_k]:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")
    except Exception as e:
        print(f"[{dataset_tag}][LogReg][Filtered] Error printing top identifiers: {e}")


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
            f"[Kaggle][NLTK NaiveBayes] Acc={accuracy_score(y_test, pred_nb):.3f}  "
            f"Prec(pos)={precision_score(y_test, pred_nb, pos_label=1):.3f}  "
            f"Rec(pos)={recall_score(y_test, pred_nb, pos_label=1):.3f}"
        )
        cm = confusion_matrix(y_test, pred_nb, labels=[0, 1])
        print(f"[Kaggle][NLTK NaiveBayes] Confusion Matrix:\n{cm}")
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
            f"[Kaggle][MultinomialNB]   Acc={accuracy_score(y_test, preds):.3f}  "
            f"Prec(pos)={precision_score(y_test, preds, pos_label=1):.3f}  "
            f"Rec(pos)={recall_score(y_test, preds, pos_label=1):.3f}"
        )
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f"[Kaggle][MultinomialNB] Confusion Matrix:\n{cm}")
    except Exception as e:
        print(f"[MultinomialNB] Error: {e}")

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
        preds = pred_lr.tolist() if hasattr(pred_lr, "tolist") else pred_lr
        print(
            f"[Kaggle][LogReg]         Acc={accuracy_score(y_test, preds):.3f}  "
            f"Prec(pos=clickbait)={precision_score(y_test, preds, pos_label=1):.3f}  "
            f"Rec(pos=clickbait)={recall_score(y_test, preds, pos_label=1):.3f}"
        )
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f"[Kaggle][LogReg]         Confusion Matrix:\n{cm}")
        print_top_lr_identifiers(lr_pipeline, "Kaggle", top_k=10)
        print_top_lr_identifiers_filtered(lr_pipeline, "Kaggle", top_k=10)
    except Exception as e:
        print(f"[Kaggle][LogReg] Error: {e}")


# ================= Train2 Naive Bayes + MultinomialNB (no shuffle) =================
def run_train2_experiments():
    print("\n=== Train2 Dataset ===")
    X_texts, y = load_train2_texts_labels()
    if not X_texts or not y:
        print("No data loaded from Train2 file; skipping experiments.")
        return

    # Deterministic contiguous 80/20 split, no shuffle
    n = len(X_texts)
    split = max(1, int(0.8 * n))
    X_train, X_test = X_texts[:split], X_texts[split:]
    y_train, y_test = y[:split], y[split:]

    # NLTK Naive Bayes with boolean presence features over top-N vocab
    vocab = build_vocab_kaggle(X_train)
    def featurize(texts):
        return [document_features_kaggle(regex_tokenize(t), vocab) for t in texts]

    train_set_t2 = list(zip(featurize(X_train), y_train))
    test_feats_t2 = featurize(X_test)

    # Train and evaluate NLTK Naive Bayes
    try:
        from nltk.classify import NaiveBayesClassifier as NLTKNaiveBayes
        nb_t2 = NLTKNaiveBayes.train(train_set_t2)
        pred_nb = [nb_t2.classify(x) for x in test_feats_t2]
        print(
            f"[Train2][NLTK NaiveBayes] Acc={accuracy_score(y_test, pred_nb):.3f}  "
            f"Prec(pos)={precision_score(y_test, pred_nb, pos_label=1):.3f}  "
            f"Rec(pos)={recall_score(y_test, pred_nb, pos_label=1):.3f}"
        )
        cm = confusion_matrix(y_test, pred_nb, labels=[0, 1])
        print(f"[Train2][NLTK NaiveBayes] Confusion Matrix:\n{cm}")
    except Exception as e:
        print(f"[Train2][NLTK NaiveBayes] Error: {e}")

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
        preds = pred_mnb.tolist() if hasattr(pred_mnb, "tolist") else pred_mnb
        print(
            f"[Train2][MultinomialNB]   Acc={accuracy_score(y_test, preds):.3f}  "
            f"Prec(pos)={precision_score(y_test, preds, pos_label=1):.3f}  "
            f"Rec(pos)={recall_score(y_test, preds, pos_label=1):.3f}"
        )
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f"[Train2][MultinomialNB] Confusion Matrix:\n{cm}")
    except Exception as e:
        print(f"[Train2][MultinomialNB] Error: {e}")

    # Logistic Regression with TF-IDF (n-grams), no shuffle split (contiguous)
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
        preds = pred_lr.tolist() if hasattr(pred_lr, "tolist") else pred_lr
        print(
            f"[Train2][LogReg]         Acc={accuracy_score(y_test, preds):.3f}  "
            f"Prec(pos=clickbait)={precision_score(y_test, preds, pos_label=1):.3f}  "
            f"Rec(pos=clickbait)={recall_score(y_test, preds, pos_label=1):.3f}"
        )
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f"[Train2][LogReg]         Confusion Matrix:\n{cm}")
        print_top_lr_identifiers(lr_pipeline, "Train2", top_k=10)
        print_top_lr_identifiers_filtered(lr_pipeline, "Train2", top_k=10)
    except Exception as e:
        print(f"[Train2][LogReg] Error: {e}")


if __name__ == "__main__":
    # Kaggle dataset
    c1, c0 = load_data()
    print(f"[Kaggle] counts - clickbait (label=1): {c1}")
    print(f"[Kaggle] counts - news      (label=0): {c0}")
    run_kaggle_experiments()

    # Train2 dataset (no shuffle)
    t1, t0 = load_data_train2()
    print(f"[Train2] counts - clickbait (label=1): {t1}")
    print(f"[Train2] counts - news      (label=0): {t0}")
    run_train2_experiments()
