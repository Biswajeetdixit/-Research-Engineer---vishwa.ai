# main.py

import logging
import argparse
from dataloader_module import DataLoader

from model import SummarizationModel
from pipeline import SummarizationPipeline
from evaluator import Evaluator
from transformers import AutoTokenizer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    setup_logging()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataloader = DataLoader(tokenizer_name=args.model_name)
    model = SummarizationModel(model_name=args.model_name)
    pipeline = SummarizationPipeline(dataloader, model, tokenizer)

    results = pipeline.run_inference(sample_count=args.sample_count)
    
    evaluator = Evaluator()
    scores = evaluator.evaluate(results)

    for idx, item in enumerate(results):
        print(f"\nSample {idx + 1}")
        print("Dialogue:", item['dialogue'])
        print("Reference:", item['reference'])
        print("Generated:", item['generated'])

    print("\nEvaluation Scores:", scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--sample_count', type=int, default=5)
    args = parser.parse_args()
    main(args)
