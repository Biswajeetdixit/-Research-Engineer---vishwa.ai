Insights and Observations
Project: Modular Dialogue Summarization Pipeline
Model Used: facebook/bart-base
Dataset: SAMSum

1. Dataset Observations
Nature of the Data:
The SAMSum dataset consists of informal, multi-turn dialogues.

Most conversations are short (5–10 turns) and involve 2 speakers.

Summaries are concise, typically 1–3 sentences, written in a narrative tone.

Strengths:
Great for fast experimentation due to small size and focused domain.

Human-written summaries offer a strong reference baseline.

Challenges:
Informal language (e.g., slang, abbreviations) can confuse the model.

Pronoun resolution is difficult without context (e.g., who “he” refers to).

Some conversations have minimal context for summarization.

2. Model Behavior & Insights
Strengths of facebook/bart-base:
Handles short dialogue summarization reasonably well without fine-tuning.

Maintains important keywords (meeting, deadlines, reminder).

Can rephrase the conversation effectively in a summary form.

Common Issues:
Redundancy: Sometimes repeats ideas (e.g., “he said he might join... he said he’d like to join”).

Hallucination: Occasionally introduces statements not in the dialogue.

Cut-off Summaries: When not properly constrained, summary generation may end abruptly (e.g., "Alright, I...").

Observations:
The model performs better when dialogues are slightly cleaned (removing fillers or irrelevant exchanges).

Sensitive to input length and formatting — adding speaker tags helps.

ROUGE scores provide a basic evaluation, but qualitative checks are essential.

3. Evaluation Summary
ROUGE-L and ROUGE-1 were generally higher than ROUGE-2 (consistent with dialogue summarization research).

In most samples, the generated summaries preserved intent but not always exact phrasing.

4. Limitations & Future Improvements
Limitations:
No fine-tuning — using a pre-trained model out-of-the-box.

No custom post-processing or dialogue restructuring applied.

Summary truncation is a risk without careful token limits.

Improvements to Explore:
Fine-tune BART or T5 on SAMSum for better performance.

Add co-reference resolution or speaker tagging preprocessing.

Use multiple models (e.g., t5-small, pegasus) for comparison.

Visualize token importance or attention weights for better explainability.