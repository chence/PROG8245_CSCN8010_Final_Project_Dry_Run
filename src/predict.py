from __future__ import annotations

import argparse
import os
from typing import Dict

from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI

from src.data_processing import DEFAULT_RESPONSES_PATH, load_artifact, load_response_templates

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return 'en'


def translate_text(text: str, dest: str) -> str:
    if client is None:
        return text
    try:
        # Map language codes to full language names for better translation
        language_map = {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
        }
        dest_lang = language_map.get(dest, dest)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Translate the following text to {dest_lang}. Return only the translated text without any explanation."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


def predict_intent(text: str, model_path: str = 'models/baseline_nb.joblib', responses_path: str = str(DEFAULT_RESPONSES_PATH)) -> Dict[str, str]:
    model = load_artifact(model_path)
    responses = load_response_templates(responses_path)

    language = detect_language(text)
    english_text = translate_text(text, 'en') if language != 'en' else text
    intent = model.predict([english_text])[0]
    english_response = responses.get(intent, 'I can only support basic non-emergency medical information at the moment.')
    final_response = translate_text(english_response, language) if language != 'en' else english_response

    return {
        'language': language,
        'english_text': english_text,
        'intent': intent,
        'response': final_response,
        'english_response': english_response,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a MediChat intent label from text.')
    parser.add_argument('text', help='Input text for prediction.')
    parser.add_argument('--model-path', default='models/baseline_nb.joblib')
    args = parser.parse_args()

    result = predict_intent(args.text, model_path=args.model_path)
    for key, value in result.items():
        print(f'{key}: {value}')
