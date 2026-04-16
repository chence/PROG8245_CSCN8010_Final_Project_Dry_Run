from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path('data/raw/medical_intent_dataset.csv')
DEFAULT_RESPONSES_PATH = Path('data/raw/intent_responses.json')


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'text', 'label'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = df.copy()
    df['text'] = df['text'].astype(str).map(clean_text)
    df = df.dropna(subset=['text', 'label']).drop_duplicates().reset_index(drop=True)
    return df


def load_response_templates(path: str | Path = DEFAULT_RESPONSES_PATH) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_dataset(df: pd.DataFrame, test_size: float = 0.25):
    return train_test_split(
        df['text'],
        df['label'],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df['label'],
    )


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_artifact(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path):
    return joblib.load(path)
