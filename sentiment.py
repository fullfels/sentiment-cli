"""Train and use a compact sentiment classifier from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


DEFAULT_DATA = Path(__file__).parent / "data" / "reviews.csv"


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=6000,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
        ]
    )


def train(data_path: Path = DEFAULT_DATA) -> tuple[Pipeline, dict[str, float]]:
    data = pd.read_csv(data_path)
    model = build_model()
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(model, data["text"], data["label"], cv=folds, scoring="accuracy")
    model.fit(data["text"], data["label"])
    metrics = {"cv_accuracy_mean": round(float(scores.mean()), 3), "cv_accuracy_std": round(float(scores.std()), 3)}
    return model, metrics


def predict(model: Pipeline, text: str) -> dict[str, object]:
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return {"label": str(model.classes_[index]), "confidence": round(float(probabilities[index]), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--model", type=Path, default=Path("artifacts/sentiment.joblib"))
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("text")
    predict_parser.add_argument("--model", type=Path, default=Path("artifacts/sentiment.joblib"))
    args = parser.parse_args()

    if args.command == "train":
        model, metrics = train(args.data)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model)
        print(json.dumps(metrics, indent=2))
    else:
        model = joblib.load(args.model)
        print(json.dumps(predict(model, args.text), indent=2))


if __name__ == "__main__":
    main()
