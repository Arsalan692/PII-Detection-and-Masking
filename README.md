# PII Detection and Masking

This repository contains a comparative study and implementation of Personal Identifiable Information (PII) detection, specifically targeting Person Names (`PER`) and Email Addresses (`EMAIL`). It evaluates the performance of a fine-tuned **RoBERTa-base** token classification model against a zero-shot **LLaMA-3.2-1B-Instruct** large language model.

## Overview

Detecting PII is a critical step in data privacy and security workflows. This project utilizes the WikiNeural dataset, augmented with synthetically injected email addresses, to test the models' capabilities in finding exact boundary spans for PII.

### Key Findings
* **Fine-Tuned RoBERTa** achieved a **0.95 F1 score** for email detection and **0.93 F1 score** for person name detection.
* **Zero-Shot LLaMA 1B** struggled significantly without task-specific constraints, managing only a **0.42 F1 score** for email detection and suffering from a high hallucination rate (inventing fake emails in 41.8% of cases where none existed).
* **Conclusion**: For production PII detection, fine-tuned token classification models remain vastly superior to zero-shot LLMs in terms of accuracy, boundary precision, speed, and resource efficiency.

---

## Repository Structure

The pipeline is split into five distinct Python scripts, meant to be run sequentially:

* **`Data_preparation.py`**  
  Augments the base WikiNeural dataset by generating realistic email addresses from person names already present in the text. It injects these emails into 15% of the entries using different formatting strategies (normal and spaced) and disjoint sentence templates for train/test splits to avoid data leakage.

* **`Tokenize_and_Split.py`**  
  Loads the prepared NER data, performs a stratified split (90% train / 10% validation) to preserve the proportion of email-containing entries, and tokenizes the text using `RobertaTokenizerFast`. Importantly, it propagates the `I-EMAIL` label to RoBERTa sub-tokens to ensure proper learning of multi-token email addresses.

* **`Fine_tuning.py`**  
  Fine-tunes the `roberta-base` model for token classification. Optimized for a 16GB T4 GPU using FP16, gradient accumulation, and gradient checkpointing. It saves the best checkpoint based on validation F1 scores.

* **`LLaMa.py`**  
  Runs `meta-llama/Llama-3.2-1B-Instruct` in zero-shot mode on a stratified 325-example subset of the test data. Evaluates how well the LLM identifies names and emails without any task-specific training, using a pure instruction-based prompt.

* **`Final_evaluation.py`**  
  The ultimate evaluation script. It loads the best fine-tuned RoBERTa checkpoint and runs it against the completely held-out test set. It computes span-level metrics (seqeval), token-level FPR/FNR, and performs an extensive error analysis (highlighting False Positives and False Negatives).

---

## Results & Performance

| Metric | RoBERTa (fine-tuned) | LLaMA 1B (zero-shot) | Delta |
| :--- | :--- | :--- | :--- |
| **PER F1** | 0.9307 | 0.8529 | +0.0778 |
| **EMAIL F1** | 0.9492 | 0.4183 | +0.5309 |
| **Hallucinations** | 0 | 136/325 (41.8%) | — |

*Detailed insights and error analysis can be found in the included `Report.pdf`.*

---

## Setup & Requirements

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arsalan692/PII-Detection-and-Masking.git
   cd PII-Detection-and-Masking
   ```

2. **Install Dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install transformers datasets seqeval scikit-learn torch huggingface_hub
   ```

3. **Hugging Face Authentication (For LLaMA):**
   If you plan to run the LLaMA zero-shot script, you must authenticate with Hugging Face using an access token:
   * Generate an access token on your Hugging Face account.
   * Add the token in `LLaMa.py` on line 28: `login(token="YOUR_HUGGING_FACE_TOKEN")`.

## License & Acknowledgements
This project was developed as a case study for PII extraction mechanisms. The dataset used builds upon the WikiNeural dataset.
