# Development Plan

## Introduction
MediChat is a team-based academic chatbot project focused on multilingual and audio/text medical question support. The project will be developed in small, testable stages so that the team can demonstrate progress early and reduce integration risk.

## Use Cases

### Use Case 1 - Symptom Inquiry
A user types: `I have a headache and fever. What could it be?`
The system classifies the question as a symptom-related intent and returns general information.

### Use Case 2 - Self-Care Advice
A user asks: `What should I do for a sore throat?`
The system returns common self-care guidance such as hydration, rest, and monitoring symptoms.

### Use Case 3 - Medication Question
A user asks: `Can I take cough syrup for a dry cough?`
The chatbot provides general educational information and reminds the user to check professional guidance.

### Use Case 4 - Multilingual Interaction
A user asks in Chinese or Spanish. The system detects the language, translates the question, predicts the intent, and translates the response back.

## Development Roadmap

### Phase 1 - Repository Setup
- create the project folder structure
- initialize Git and DVC
- upload to GitHub
- keep the original prototype notebook in `notebooks/`

### Phase 2 - Baseline Model
- build a small curated dataset
- train TF-IDF + Naive Bayes
- support command-line or simple web testing

### Phase 3 - Multilingual Support
- detect user language
- translate non-English input into English
- translate output back to the original language

### Phase 4 - Real Medical Dataset
- integrate the HPAI-BSC medical-specialities dataset
- clean the data
- retrain the baseline and comparison models

### Phase 5 - Evaluation
- produce confusion matrices
- compare precision, recall, and F1-score
- identify difficult intents and error patterns

### Phase 6 - Application Layer
- build a Gradio or Streamlit web page
- add audio input support
- improve user experience and response formatting

### Phase 7 - Model Comparison
- evaluate baseline Naive Bayes
- evaluate SVD + Logistic Regression
- evaluate PCA + Logistic Regression
- visualize model comparison results

## Team Deliverables
- working repository with DVC tracking
- prototype notebook archive
- runnable source code
- baseline evaluation outputs
- pitch, architecture, and development documentation
