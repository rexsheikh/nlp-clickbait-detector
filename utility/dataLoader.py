# Utilities
    # - dataloaders for csv and jsonl format data

import csv
import json
from pathlib import Path


# Data paths (see readme for details on project file structure)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

KAGGLE_DATASET_PATH = str(DATA_DIR / "kaggle_clickbait.csv")
TRAIN2_DATASET_PATH = str(DATA_DIR / "news_clickbait_dataset" / "train2.csv")
WEBIS_INSTANCES_PATH = str(DATA_DIR / "webis-data" / "instances.jsonl")
WEBIS_TRUTH_PATH = str(DATA_DIR / "webis-data" / "truth.jsonl")


# loads kaggle dataset, both in csv format. 
# params: path (str or None)
# return: texts (list[str]), labels (list[int])
def load_kaggle_texts_labels(path=None):
    path = path or KAGGLE_DATASET_PATH
    texts, labels = [], []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("clickbait")
            text = row.get("headline")
            # skip empty rows and clean
            if raw is None or text is None:
                continue
            y = int(str(raw).strip().lower())
            texts.append(str(text).strip())
            labels.append(y)
    return texts, labels


# Train2 loader (label{"news","clickbait"}, title); BOM-safe.
# params: path (str or None)
# return: texts (list[str]), labels (list[int: 1=clickbait, 0=news])
def load_train2_texts_labels(path=None):
    path = path or TRAIN2_DATASET_PATH
    texts, labels = [], []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("label")
            text = row.get("title")
            if raw is None or text is None:
                continue
            label = str(raw).strip().lower()
            if label not in {"clickbait", "news"}:
                continue
            y = 1 if label == "clickbait" else 0
            texts.append(str(text).strip())
            labels.append(y)
    return texts, labels


# Internal: map Webis truthClass to int (clickbait->1, else 0).
def _webis_label_from_truth_class(truth_class):
    if truth_class is None:
        return 0
    return 1 if str(truth_class).strip().lower() == "clickbait" else 0


# Webis loader (join truth.jsonl and instances.jsonl on id); BOM-safe ids/text.
# params: instances_path (str or None), truth_path (str or None)
# return: texts (list[str]), labels (list[int])
def load_webis_texts_labels(instances_path=None, truth_path=None):
    instances_path = instances_path or WEBIS_INSTANCES_PATH
    truth_path = truth_path or WEBIS_TRUTH_PATH
    truth_map = {}
    texts, labels = [], []

    # Load truth (id -> label)
    with open(truth_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = (raw or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = str(obj.get("id", "")).lstrip("\ufeff")
            truth_class = obj.get("truthClass", "")
            truth_map[tid] = _webis_label_from_truth_class(truth_class)

    # Load instances and join
    with open(instances_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = (raw or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(obj.get("id", "")).lstrip("\ufeff")
            if iid not in truth_map:
                continue
            post_text_list = obj.get("postText") or []
            text = ""
            if isinstance(post_text_list, list) and len(post_text_list) > 0:
                # prefer first, fallback to join
                text = str(post_text_list[0]).lstrip("\ufeff").strip()
                if not text and len(post_text_list) > 1:
                    text = " ".join(map(str, post_text_list)).strip()
            if not text:
                continue
            texts.append(text)
            labels.append(truth_map[iid])

    return texts, labels


# Unified selector for convenience.
# params: dataset in {"kaggle","train2","webis"}
# overrides: path=..., instances_path=..., truth_path=...
# return: texts, labels
def load_texts_labels(dataset, **overrides):
    ds = str(dataset).lower().strip()
    if ds == "kaggle":
        return load_kaggle_texts_labels(overrides.get("path"))
    if ds == "train2":
        return load_train2_texts_labels(overrides.get("path"))
    if ds == "webis":
        return load_webis_texts_labels(
            instances_path=overrides.get("instances_path"),
            truth_path=overrides.get("truth_path"),
        )
    raise ValueError(f"Unknown dataset: {dataset}")


# Convenience: counts for a dataset via unified loader.
# return: dict {"clickbait": c1, "news": c0}
def load_counts(dataset, **overrides):
    _, labels = load_texts_labels(dataset, **overrides)
    c1 = sum(1 for v in labels if v == 1)
    c0 = sum(1 for v in labels if v == 0)
    return {"clickbait": c1, "news": c0}


if __name__ == "__main__":
    # Quick self-check (non-failing if files absent)
    try:
        for ds in ["kaggle", "train2", "webis"]:
            try:
                X, y = load_texts_labels(ds)
                print(f"[{ds}] loaded rows: {len(X)}  pos={sum(y)} neg={len(y)-sum(y)}")
            except Exception as e:
                print(f"[{ds}] loader error: {e}")
    except Exception as e:
        print(f"[self-check] error: {e}")
