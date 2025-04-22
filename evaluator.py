# evaluator.py

import logging
from evaluate import load

class Evaluator:
    def __init__(self):
        self.rouge = load("rouge")
        logging.info("Evaluator with ROUGE metric initialized.")

    def evaluate(self, predictions):
        refs = [item['reference'] for item in predictions]
        gens = [item['generated'] for item in predictions]
        scores = self.rouge.compute(predictions=gens, references=refs)
        return scores

