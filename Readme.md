

## Project structure
```
nlp-clickbait-detector/
├── .gitignore
├── classical/
│   ├── logisticRegression.py
│   ├── naiveBayes.py
│   └── svm.py
├── data/
├── deliverables/
│   ├── nlp - final project - presentation.pdf
│   ├── nlp_final_report.pdf
│   ├── presRecordingLink.txt
│   └── report.tex
├── gen_metrics.py
├── plots/
│   ├── classical_metrics.png
│   ├── f1_scatter.png
│   └── scatter_by_model/
│       ├── accuracy_LogReg.png
│       ├── accuracy_NaiveBayes.png
│       ├── accuracy_SVM-SGD.png
│       ├── macro_f1_LogReg.png
│       ├── macro_f1_NaiveBayes.png
│       ├── macro_f1_SVM-SGD.png
│       ├── precision_LogReg.png
│       ├── precision_NaiveBayes.png
│       ├── precision_SVM-SGD.png
│       ├── recall_LogReg.png
│       ├── recall_NaiveBayes.png
│       └── recall_SVM-SGD.png
├── raw-outputs/
│   ├── logisticRegressionRaw.txt
│   ├── metrics_summary.csv
│   ├── naiveBayesRaw.txt
│   ├── presentation.txt
│   ├── svmRaw.txt
│   └── to-do.txt
├── transformers_notebook.ipynb
├── utility/
│   ├── common_text.py
│   └── dataLoader.py
└── Readme.md
```

## Setup
- Install dependencies:
  - pip install scikit-learn numpy scipy pandas nltk
  - uncomment any nltk.download statements
  - for transformer, visit google colab, upload data/utlities directories and the transformer.ipynb. Set runtime to use t4 gpu. If local, mirror the above approach.  

## Datasets
- Place dataset files under nlp-clickbait-detector/data/.
- The loader unifies multiple sources (e.g., kaggle, train2, webis). See utility/dataLoader.py for exact filenames/format expectations.

## Quick start (classical baselines)
Run any model from within nlp-clickbait-detector/:

- Logistic Regression
  - python classical/logisticRegression.py --dataset webis --show-identifiers
- Linear SVM (SGD)
  - python classical/svm.py --dataset kaggle
- Naive Bayes
  - python classical/naiveBayes.py --dataset train2

Use --dataset all to evaluate across all available datasets.

### Useful options (all classical scripts)
- --min-df, --max-df: TF–IDF pruning thresholds
- --min-df-high: shortcut for stricter pruning (min_df=5, max_df=0.95)
- --use-stopwords: remove standard English stopwords
- --ignore-terms: filter common, low-signal terms defined in utility/common_text.py
- --punct-signals: add punctuation-based features (exclamation, question marks, emoji)
- --struct-features: add length/structure features (e.g., token/char counts)
- --C (LogReg), --class-weight: regularization strength and balancing
- --show-identifiers: print top positive/negative weighted features

Examples:
- python classical/logisticRegression.py --dataset all --min-df-high --use-stopwords --ignore-terms --show-identifiers
- python classical/svm.py --dataset webis --min-df 3 --max-df 0.9

## Metrics and plots
- After running models, you can aggregate results and generate plots:
  - python gen_metrics.py
- Summary CSV: raw-outputs/metrics_summary.csv
- Figures: plots/ and plots/scatter_by_model/
