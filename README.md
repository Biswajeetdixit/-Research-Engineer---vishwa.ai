# Dialogue Summarization Pipeline

## Overview

This project implements a modular, class-based dialogue summarization pipeline using a self-hosted transformer model (`facebook/bart-base`). The goal is to summarize dialogues, such as those in the SAMSum dataset, using a locally-run model without relying on external APIs like OpenAI.

The project is structured into separate components for data loading, model inference, evaluation, and a user-friendly Streamlit interface.

## Features

- Modular architecture using Python classes
- Supports any Hugging Face-compatible summarization model
- Uses the SAMSum dialogue summarization dataset
- ROUGE-based evaluation of generated summaries
- Streamlit app for interactive testing and evaluation
- CLI runner for quick experiments and batch evaluation

## Project Structure
```
├── app.py                  # Streamlit UI for manual input and summary generation
├── dataloader_module.py   # Data loading and tokenization
├── evaluator.py           # ROUGE-based evaluation class
├── model.py               # Summarization model wrapper (BART)
├── pipeline.py            # Pipeline for running inference across samples
├── main.py                # Command-line interface for batch processing
├── requirements.txt       # Python dependencies

```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone
https://github.com/Biswajeetdixit/-Research-Engineer---vishwa.ai

```
### 2. Create Virtual Environment (Optional)

```
cd C:\Users\biswa\Desktop\Assignment_for_companyes\Vishwa.ai
python -m venv biswajeet
biswajeet\Scripts\activate
```


### 3. Install Requirements
```
pip install -r requirements.txt
```



## Running the Application
### Option 1: Streamlit Interface

streamlit run app.py
Paste a dialogue in the text box

Click "Generate Summary"

(Optional) Paste a reference summary to compute ROUGE score

## Option 2: Command-Line Evaluation

python main.py --model_name facebook/bart-base --sample_count 5
## Model
facebook/bart-base is used as the base summarization model.

You can easily replace it with other Hugging Face models like t5-small, pegasus-xsum, etc.

## Dataset
Dataset: SAMSum

Description: A collection of short dialogues paired with human-written summaries.

## Evaluation
Evaluation metric: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Metric is calculated using the Hugging Face evaluate package

## Notes
The code uses no external APIs — all models are hosted and run locally

All scripts are written with extensibility and clarity in mind
