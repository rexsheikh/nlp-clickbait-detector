

## Project structure
```
nlp-clickbait-detector/
├── classical/
│   ├── logisticRegression.py     
│   ├── naiveBayes.py            
│   └── svm.py                   
├── utility/
│   ├── common_text.py            
│   └── dataLoader.py            
├── data/                         # place datasets here (reference slides for source links and adjust paths as necessary to load properly in dataLoader.py)
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
├── gen_metrics.py                
├── metrics_summary.csv          
├── logisticRegressionRaw.txt     # from piping output to .txt (optional, recommended if tweaking code, metrics or printouts, or experiments for all these raw outputs)
├── naiveBayesRaw.txt            
├── svmRaw.txt                    
├── presentation.txt              
├── report.tex                   
├── transformers_notebook.ipynb  
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
- Summary CSV: metrics_summary.csv
- Figures: plots/ and plots/scatter_by_model/
