# pipeline.py

import logging

class SummarizationPipeline:
    def __init__(self, dataloader, model, tokenizer):
        self.dataloader = dataloader
        self.model = model
        self.tokenizer = tokenizer
        logging.info("Pipeline initialized.")

    def run_inference(self, sample_count=5):
        dataset = self.dataloader.load()
        summaries = []
        for sample in dataset.select(range(sample_count)):
            generated = self.model.summarize(self.tokenizer, sample['dialogue'])
            summaries.append({
                'dialogue': sample['dialogue'],
                'reference': sample['summary'],
                'generated': generated
            })
        return summaries
