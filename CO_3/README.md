# Advanced NLP Lab: Transformers & Sequence Modeling (CO3)

A collection of hands-on experiments covering transformer-based NLP using Hugging Face, plus a document-processing pipeline that combines classical ML with probabilistic and finite-state modeling. Every experiment is written as a **Google Colab-friendly notebook** that uses small data subsets so it runs in a few minutes on a free T4 GPU.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Colab](https://img.shields.io/badge/Google-Colab-orange)

---

## Experiments

| No. | Experiment | Model(s) | Dataset | CO |
|-----|-----------|----------|---------|----|
| 24 | Fine-Tuning BERT for Text Classification | `bert-base-uncased` | IMDB | CO3 |
| 25 | Sentence Embeddings using BERT and GPT-2 | `bert-base-uncased`, `gpt2` | Sample sentence pairs | CO3 |
| 26 | Text Generation using GPT-2 on a Custom Dataset | `gpt2` / `distilgpt2` | Custom `.txt` corpus | CO3 |
| 27 | Named Entity Recognition using Transformers (BERT) | `bert-base-cased` | CoNLL-2003 | CO3 |
| 28 | Machine Translation using MarianMT or T5 | `Helsinki-NLP/opus-mt-*`, `t5-small` | OPUS / WMT subset | CO3 |
| 29 | Text Summarization using BART or T5 | `facebook/bart-large-cnn`, `t5-small` | CNN/DailyMail | CO3 |
| 30 | Chatbot using a Pre-trained Transformer | `microsoft/DialoGPT-medium` | N/A (pre-trained) | CO3 |
| 31 | Automatic Document Classification and Separation using ML | TF-IDF + Logistic Regression / SVM | 20 Newsgroups / custom pages | CO3 |
| 32 | Hybrid Probabilistic + FSM-Based Sequence Modeling for Document Boundary Detection | Probabilistic classifier + FSM / Viterbi | Page sequences from Exp. 31 | CO3 |

---

## Experiment Details

### 24. Fine-Tuning BERT for Text Classification
Fine-tunes `bert-base-uncased` for binary sentiment classification using the Hugging Face `Trainer` API.
- 4,000 train / 1,000 test subset, `max_length=128`, mixed precision (`fp16`), 2 epochs
- Metrics: accuracy, F1
- Includes inference with `pipeline("text-classification")` and model saving

### 25. Sentence Embeddings using BERT and GPT-2
Extracts fixed-size sentence vectors from both an encoder (BERT) and a decoder (GPT-2).
- BERT: mean pooling over token embeddings (using the attention mask)
- GPT-2: mean pooling (or last-token pooling) of hidden states
- Compares sentence similarity with cosine similarity and visualizes embeddings with PCA / t-SNE

### 26. Text Generation using GPT-2 on a Custom Dataset
Fine-tunes GPT-2 as a causal language model on your own text file.
- Tokenization and block grouping with `DataCollatorForLanguageModeling`
- Generation with `temperature`, `top_k`, `top_p`, and `repetition_penalty`
- Reports perplexity before and after fine-tuning

### 27. Named Entity Recognition using BERT
Token classification (PER, ORG, LOC, MISC) with `AutoModelForTokenClassification`.
- Sub-word label alignment using `word_ids()`
- Evaluation with `seqeval` (precision, recall, F1)
- Inference with `pipeline("ner", aggregation_strategy="simple")`

### 28. Machine Translation using MarianMT or T5
Translates text with pre-trained seq2seq models, with optional fine-tuning.
- MarianMT: `Helsinki-NLP/opus-mt-en-hi` (or `en-fr`, `en-de`)
- T5: `t5-small` with the `translate English to German:` prefix
- Evaluation with BLEU (`sacrebleu`)

### 29. Text Summarization using BART or T5
Abstractive summarization of news articles.
- BART: `facebook/bart-large-cnn`; T5: `t5-small` (`summarize:` prefix)
- Beam search with length controls (`max_length`, `min_length`, `num_beams`)
- Evaluation with ROUGE-1 / ROUGE-2 / ROUGE-L

### 30. Chatbot using DialoGPT
Multi-turn conversational agent using `microsoft/DialoGPT-medium`.
- Maintains chat history by concatenating previous turns
- Sampling-based decoding (`top_k`, `top_p`, `temperature`) for varied replies
- Simple console loop (or Gradio UI) for interactive chat

### 31. Automatic Document Classification and Separation using ML
Classifies each page of a mixed, scanned or digital batch into a document type, then separates the batch into individual documents.
- Features: TF-IDF (word + character n-grams)
- Classifiers: Logistic Regression, Linear SVM, Naive Bayes (compared)
- Separation: a new document starts when the predicted class changes or confidence drops
- Metrics: accuracy, macro-F1, confusion matrix

### 32. Hybrid Probabilistic + FSM-Based Sequence Modeling for Document Boundary Detection
Improves page-level separation by modeling the *sequence* of pages rather than each page in isolation.
- **Probabilistic layer:** a classifier outputs per-page probabilities for `NEW_DOCUMENT` vs `CONTINUATION`, along with class likelihoods
- **FSM layer:** states `START → IN_DOCUMENT → BOUNDARY → IN_DOCUMENT …` with allowed/forbidden transitions (e.g., a page cannot continue a document that has not started)
- **Decoding:** Viterbi decoding combines emission probabilities with transition probabilities to find the most likely boundary sequence
- Metrics: boundary precision, recall, F1, and comparison against the classifier-only baseline

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── 24_bert_text_classification.ipynb
├── 25_sentence_embeddings_bert_gpt2.ipynb
├── 26_gpt2_text_generation.ipynb
├── 27_bert_ner.ipynb
├── 28_machine_translation.ipynb
├── 29_text_summarization.ipynb
├── 30_dialogpt_chatbot.ipynb
├── 31_document_classification_separation.ipynb
├── 32_hybrid_probabilistic_fsm_boundary_detection.ipynb
└── data/
    └── custom_corpus.txt        # used in Experiment 26
```

---

## Getting Started

### Run on Google Colab (recommended)
1. Open the notebook in Colab (File → Upload notebook, or use "Open in Colab" from GitHub).
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells.

### Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

### `requirements.txt`
```
transformers
datasets
evaluate
accelerate
torch
scikit-learn
seqeval
sacrebleu
rouge_score
sentencepiece
sacremoses
numpy
pandas
matplotlib
tensorflow
```

---

## Tips

- **Faster runs:** reduce the dataset subset size, lower `max_length`, or switch to a smaller model such as `distilbert-base-uncased` or `t5-small`.
- **Better accuracy:** use more training samples, more epochs, or longer sequences.
- **Dataset errors:** newer `huggingface_hub` versions need full repo ids, e.g. `load_dataset("stanfordnlp/imdb")` rather than `load_dataset("imdb")`.
- **Out of memory:** lower `per_device_train_batch_size` and keep `fp16=True`.

---

## Tech Stack

Python · PyTorch · TensorFlow/Keras · Hugging Face Transformers & Datasets · scikit-learn · Google Colab

---

## Author

Gaurav Sahu
Course / Roll No: `2401331520073`
GitHub: [@your-username](https://github.com/gauravv012)



