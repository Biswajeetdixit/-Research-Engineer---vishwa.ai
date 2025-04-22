





import streamlit as st
from dataloader_module import DataLoader
from model import SummarizationModel
from evaluator import Evaluator
from transformers import AutoTokenizer

# Initialize model and tokenizer
MODEL_NAME = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SummarizationModel(MODEL_NAME)
evaluator = Evaluator()

st.set_page_config(page_title="Dialogue Summarizer", layout="wide")
st.title("🗣️ Dialogue Summarization with BART")

# Text input
st.subheader("Enter a conversation or dialogue:")
dialogue_input = st.text_area("Paste your dialogue here:", height=250)

if st.button("Generate Summary"):
    if dialogue_input.strip():
        with st.spinner("Generating summary..."):
            summary = model.summarize(tokenizer, dialogue_input)

            st.subheader("📝 Generated Summary:")
            st.success(summary)

            if st.checkbox("Evaluate against a reference summary?"):
                reference = st.text_area("Paste the reference summary here:")
                if reference:
                    rouge_score = evaluator.evaluate([{
                        'reference': reference,
                        'generated': summary
                    }])
                    st.subheader("📊 ROUGE Evaluation:")
                    st.json(rouge_score)
    else:
        st.warning("Please enter some dialogue.")
