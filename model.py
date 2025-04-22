# model.py

import logging
from transformers import AutoModelForSeq2SeqLM

class SummarizationModel:
    def __init__(self, model_name='facebook/bart-base'):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logging.info(f"Loaded model {model_name}")

    def summarize(self, tokenizer, text, max_length=128):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=max_length, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
