import logging
from datasets import load_dataset
from transformers import AutoTokenizer


# from dataloader_module import DataLoader

class DataLoader:
    def __init__(self, dataset_name='samsum', split='train', tokenizer_name='facebook/bart-base', max_length=1024):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        logging.info(f"Initialized DataLoader with dataset: {dataset_name}, split: {split}")

    def load(self):
        dataset = load_dataset(self.dataset_name)[self.split]
        logging.info(f"Loaded {len(dataset)} samples from {self.dataset_name}")
        return dataset

    def tokenize(self, dataset):
        def preprocess(sample):
            inputs = self.tokenizer(
                sample['dialogue'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    sample['summary'],
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
            return {
                'input_ids': inputs.input_ids.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': targets.input_ids.squeeze()
            }

        return dataset.map(preprocess)

