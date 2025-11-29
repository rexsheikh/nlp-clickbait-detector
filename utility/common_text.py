# Shared text helpers and constants (canonicalized from naiveBayes.py)
import re
import nltk
from nltk.corpus import stopwords

# Constants
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
    return nltk.word_tokenize(text or "")

# description: filter tokens by IGNORE_TERMS constant (case-insensitive)
# params: tokens (list[str])
# return: list[str]
def filter_ignored(tokens):
    return [w for w in tokens if w.lower() not in IGNORE_TERMS]

# description: gets stopwords if enabled is set to True (NB usage)
def get_stopwords(enabled):
    try:
        return set(stopwords.words("english")) if enabled else set()
    except Exception:
        return set()

# description: checks input text against superlative terms constant set (no regex)
# params: text (str), super_terms (set[str])
# return: bool
def contains_superlative(text, super_terms=SUPERLATIVE_TERMS):
    toks = tokenize(text or "")
    return any(t.lower() in super_terms for t in toks)

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
