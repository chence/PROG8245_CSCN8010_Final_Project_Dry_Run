from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_processing import (
    build_vectorizer,
    load_dataset,
    save_artifact,
    save_dataframe,
    split_dataset,
)

RANDOM_STATE = 42


def make_model(model_name: str):
    vectorizer = build_vectorizer()
    if model_name == 'baseline_nb':
        return Pipeline([
            ('tfidf', vectorizer),
            ('clf', MultinomialNB()),
        ])
    if model_name == 'svd_logreg':
        return Pipeline([
            ('tfidf', vectorizer),
            ('svd', TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
            ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
    if model_name == 'pca_logreg':
        return Pipeline([
            ('tfidf', vectorizer),
            ('to_dense', FunctionTransformerSparseToDense()),
            ('pca', PCA(n_components=100, random_state=RANDOM_STATE)),
            ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
    raise ValueError(f'Unsupported model_name: {model_name}')


class FunctionTransformerSparseToDense:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, 'toarray') else X



def train(model_name: str, data_path: str, model_out: str, split_out: str) -> dict:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    train_df = pd.DataFrame({'text': X_train, 'label': y_train}).reset_index(drop=True)
    test_df = pd.DataFrame({'text': X_test, 'label': y_test}).reset_index(drop=True)
    save_dataframe(train_df, Path(split_out) / 'train.csv')
    save_dataframe(test_df, Path(split_out) / 'test.csv')

    model = make_model(model_name)
    model.fit(X_train, y_train)
    save_artifact(model, model_out)

    metadata = {
        'model_name': model_name,
        'train_size': int(len(train_df)),
        'test_size': int(len(test_df)),
        'labels': sorted(df['label'].unique().tolist()),
    }
    metadata_path = Path(model_out).with_suffix('.metadata.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MediChat intent classification models.')
    parser.add_argument('--data-path', default='data/raw/medical_intent_dataset.csv')
    parser.add_argument('--model-name', choices=['baseline_nb', 'svd_logreg', 'pca_logreg'], default='baseline_nb')
    parser.add_argument('--model-out', default='models/baseline_nb.joblib')
    parser.add_argument('--split-out', default='data/processed')
    args = parser.parse_args()

    info = train(args.model_name, args.data_path, args.model_out, args.split_out)
    print(json.dumps(info, indent=2))
