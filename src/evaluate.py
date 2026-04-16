from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.data_processing import load_artifact


def evaluate(model_path: str, test_path: str, metrics_out: str, figure_out: str) -> dict:
    model = load_artifact(model_path)
    test_df = pd.read_csv(test_path)
    y_true = test_df['label']
    y_pred = model.predict(test_df['text'])

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    labels = sorted(y_true.unique())

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump({'classification_report': report, 'labels': labels, 'confusion_matrix': cm.tolist()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    fig.colorbar(im)
    fig.tight_layout()
    Path(figure_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    weighted = report['weighted avg']
    return {
        'accuracy': report['accuracy'],
        'precision_weighted': weighted['precision'],
        'recall_weighted': weighted['recall'],
        'f1_weighted': weighted['f1-score'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MediChat model performance.')
    parser.add_argument('--model-path', default='models/baseline_nb.joblib')
    parser.add_argument('--test-path', default='data/processed/test.csv')
    parser.add_argument('--metrics-out', default='documentation/evaluation_metrics.json')
    parser.add_argument('--figure-out', default='documentation/confusion_matrix.png')
    args = parser.parse_args()

    results = evaluate(args.model_path, args.test_path, args.metrics_out, args.figure_out)
    print(json.dumps(results, indent=2))
