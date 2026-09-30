# Sentiment CLI

A lightweight text-classification project that trains a TF-IDF + logistic-regression pipeline and exposes it through a simple command-line interface.

## Features

- Small built-in product-review dataset
- Character n-gram TF-IDF features that handle spelling and word variants
- Stratified cross-validation
- Saved model artifact
- CLI inference with confidence scores

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
python sentiment.py train --model artifacts/sentiment.joblib
python sentiment.py predict --model artifacts/sentiment.joblib "The setup was quick and the app works beautifully"
pytest -q
```

This example intentionally uses a tiny dataset for clarity. For production use, replace `data/reviews.csv` with a larger, domain-specific labeled corpus.

## License

MIT
