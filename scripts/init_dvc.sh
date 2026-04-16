#!/usr/bin/env bash
set -e
if [ ! -d .git ]; then
  git init
fi
if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC is not installed. Run: pip install dvc"
  exit 1
fi
dvc init
dvc add data/raw/medical_intent_dataset.csv
git add .
echo "DVC initialized. Next: git commit -m "Initialize MediChat project""
