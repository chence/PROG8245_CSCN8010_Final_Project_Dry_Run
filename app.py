from __future__ import annotations

import gradio as gr

from src.predict import predict_intent


def medichat_reply(user_text: str):
    if not user_text or not user_text.strip():
        return 'Please enter a medical question.', '', ''
    result = predict_intent(user_text)
    return result['response'], result['intent'], result['language']


with gr.Blocks(title='MediChat') as demo:
    gr.Markdown('# MediChat A multilingual educational medical chatbot for the PROG8245 final project.')
    with gr.Row():
        with gr.Column():
            user_text = gr.Textbox(label='Enter your question', lines=4, placeholder='Example: I have a sore throat and mild fever.')
            submit = gr.Button('Ask MediChat')
        with gr.Column():
            answer = gr.Textbox(label='Response', lines=6)
            intent = gr.Textbox(label='Predicted Intent')
            language = gr.Textbox(label='Detected Language')

    submit.click(medichat_reply, inputs=user_text, outputs=[answer, intent, language])


if __name__ == '__main__':
    demo.launch()
