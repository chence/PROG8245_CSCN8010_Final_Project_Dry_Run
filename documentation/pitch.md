# MediChat Pitch

## Plan and Outline Your Pitch

### Problem
People often search online for health information, but they face three common problems:
- too much information
- unreliable websites
- difficult medical language

### Solution
MediChat is a multilingual medical chatbot that gives users simple, relevant, and safer health information. Instead of opening many websites, the user can type a question and receive one focused response.

### Why This Project Matters
MediChat shows how artificial intelligence can improve access to healthcare information in a practical way. It also demonstrates how classical NLP techniques such as TF-IDF, SVD, PCA, Naive Bayes, and Logistic Regression can be used in a real chatbot pipeline.

## Sample Pitch
Many people use search engines when they feel sick, but the results are often confusing, inconsistent, and sometimes unreliable. Our project, MediChat, addresses this problem by building a narrow-domain multilingual chatbot for general respiratory health support.

MediChat takes a text question, detects the language, translates it into English when necessary, converts the question into TF-IDF features, and classifies the user's intent with machine learning models. Based on the detected intent, the system returns a simple and relevant response.

Our baseline model uses TF-IDF with Naive Bayes, and we also prepare two stronger comparison models using SVD-reduced and PCA-reduced Logistic Regression. This lets us compare performance, interpret errors with confusion matrices, and discuss how dimensionality reduction affects chatbot understanding.

The final result is a practical educational chatbot that is safe, multilingual, and easy to extend into a web and voice-based application.
