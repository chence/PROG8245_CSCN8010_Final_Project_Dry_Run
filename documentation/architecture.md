# Software Architecture

## Final Architecture Overview

```text
User Input (Text / Audio)
        ↓
Speech-to-Text Layer (future step)
        ↓
Language Detection
        ↓
Translation Layer (when input is not English)
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Optional Dimensionality Reduction (SVD or PCA)
        ↓
Intent Classification (Naive Bayes or Logistic Regression)
        ↓
Response Selection Layer
        ↓
Translation Back to User Language
        ↓
User Output (Text / Audio)
```

## Component Description

### 1. Input Layer
The chatbot accepts typed input now and is designed to support speech input later.

### 2. Language Handling Layer
The system detects the user language. If the text is not English, it translates the message to English before classification. The answer can then be translated back.

### 3. NLP Feature Layer
The text is cleaned and converted into TF-IDF vectors. This gives a sparse feature representation of the message.

### 4. Dimensionality Reduction Layer
Two optional comparison models reduce TF-IDF features into dense semantic representations:
- Truncated SVD + Logistic Regression
- PCA + Logistic Regression

### 5. Classification Layer
The baseline model uses Multinomial Naive Bayes. The advanced comparison models use Logistic Regression.

### 6. Response Layer
After the intent is predicted, the chatbot selects a prepared educational response template.

## Why This Architecture Fits the Course
This architecture directly matches the final project brief because it includes:
- TF-IDF vectorization
- optional feature compression with SVD or PCA
- Naive Bayes and Logistic Regression
- multilingual input support
- evaluation through confusion matrices and standard classification metrics
