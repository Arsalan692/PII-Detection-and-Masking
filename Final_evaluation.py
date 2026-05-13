"""
Final Evaluation on Test Set
==============================
Loads the best saved checkpoint and runs it against the held-out test set
for the first (and only) time. Computes span-level precision/recall/F1 via
seqeval, token-level FPR/FNR via sklearn, and prints a selection of false
positives and false negatives so the errors are actually interpretable.

The test set has never been touched during training or validation — this
script is the only place it gets used.

Hardware note: written for a T4 GPU (16 GB VRAM). FP16 autocast is enabled
automatically when CUDA is available.
"""

import numpy as np
from datasets import load_from_disk
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizerFast,
)
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
import json


# -----------------------------------------------------------------------------
# Label definitions
# -----------------------------------------------------------------------------

LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-EMAIL', 'I-EMAIL']
label2id   = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label   = {idx: label for idx, label in enumerate(LABEL_LIST)}

print(f"Labels: {LABEL_LIST}")
print(f"  label2id : {label2id}")
print(f"  id2label : {id2label}")


# -----------------------------------------------------------------------------
# Load model and tokenizer
# -----------------------------------------------------------------------------

DATA_DIR        = '/content/drive/MyDrive/Securiti_Internship/Internship_task_data/'
FINAL_MODEL_DIR = DATA_DIR + 'roberta_pii_final'

print(f"\nLoading model from {FINAL_MODEL_DIR} ...")

model     = RobertaForTokenClassification.from_pretrained(FINAL_MODEL_DIR)
tokenizer = RobertaTokenizerFast.from_pretrained(FINAL_MODEL_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)
model.eval()  # disable dropout for inference

use_fp16 = device.type == 'cuda'
print(f"  Running on {device}" + (" with fp16 autocast" if use_fp16 else ""))


# -----------------------------------------------------------------------------
# Load test dataset
# -----------------------------------------------------------------------------

print("\nLoading test dataset...")
test_tokenized = load_from_disk(DATA_DIR + 'test_tokenized')
print(f"  {len(test_tokenized):,} examples | columns: {test_tokenized.column_names}")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def get_predictions(dataset, model, batch_size=32):
    """Run the model over the full dataset and return aligned label sequences.

    Processes examples in batches of 32, which fits comfortably in 16 GB VRAM.
    Positions marked -100 (padding and sub-token continuations) are skipped so
    the returned sequences align one-to-one with the original word tokens.

    Returns:
        all_true : list of label-string sequences (ground truth)
        all_pred : list of label-string sequences (model output)
    """
    all_true, all_pred = [], []
    n = len(dataset)

    for start in range(0, n, batch_size):
        batch = dataset[start : start + batch_size]

        input_ids      = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        labels         = torch.tensor(batch['labels'])

        with torch.no_grad():
            if use_fp16:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = torch.argmax(outputs.logits, dim=-1).cpu()

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq, pred_seq_clean = [], []
            for pred, label in zip(pred_seq, label_seq):
                if label.item() == -100:
                    continue
                true_seq.append(id2label[label.item()])
                pred_seq_clean.append(id2label[pred.item()])
            all_true.append(true_seq)
            all_pred.append(pred_seq_clean)

        done = min(start + batch_size, n)
        if start % 320 == 0 or done == n:
            print(f"  {done:,} / {n:,} processed")

    return all_true, all_pred


print("\nRunning inference on test set...")
true_labels, pred_labels = get_predictions(test_tokenized, model)
print(f"Done — {len(true_labels):,} sequences")


# -----------------------------------------------------------------------------
# Span-level metrics (seqeval)
# -----------------------------------------------------------------------------

print("\n--- Span-level metrics (seqeval) ---")
print("A prediction counts as correct only if the entire entity span matches.\n")

report = classification_report(true_labels, pred_labels, output_dict=False)
print(report)

report_dict = classification_report(true_labels, pred_labels, output_dict=True)
print("Per-entity summary:")
for entity in ['PER', 'EMAIL']:
    if entity in report_dict:
        e = report_dict[entity]
        print(f"  {entity:8s}  P={e['precision']:.4f}  R={e['recall']:.4f}  "
              f"F1={e['f1-score']:.4f}  support={e['support']}")

# A note on how seqeval scores spans:
#   PER   -> "Maria Sharapova" is only correct if both B-PER and I-PER match
#   EMAIL -> the full token sequence (including I-EMAIL continuations) must match


# -----------------------------------------------------------------------------
# Token-level FPR / FNR (sklearn)
# -----------------------------------------------------------------------------

flat_true = [label for seq in true_labels for label in seq]
flat_pred = [label for seq in pred_labels for label in seq]
total_tokens = len(flat_true)
print(f"\n--- Token-level FPR / FNR (sklearn) ---")
print(f"Total tokens evaluated: {total_tokens:,}\n")


def compute_fpr_fnr(true, pred, target_label):
    """Compute FPR and FNR for a single label treated as a binary problem.

    Collapses the multi-class labels into: 1 if the token matches
    target_label, 0 otherwise. Then reads off the confusion matrix.

    FPR = FP / (FP + TN)  — how often non-entity tokens get wrongly flagged
    FNR = FN / (FN + TP)  — how often real entity tokens get missed

    Returns fpr, fnr, and the raw TP/TN/FP/FN counts.
    """
    binary_true = [1 if l == target_label else 0 for l in true]
    binary_pred = [1 if p == target_label else 0 for p in pred]

    cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return fpr, fnr, int(tp), int(tn), int(fp), int(fn)


header = f"  {'Label':12s}  {'FPR':>8}  {'FNR':>8}  {'TP':>7}  {'FP':>7}  {'FN':>7}  {'TN':>9}"
print(header)
print(f"  {'-' * (len(header) - 2)}")

for label in ['B-PER', 'I-PER', 'B-EMAIL', 'I-EMAIL']:
    fpr, fnr, tp, tn, fp, fn = compute_fpr_fnr(flat_true, flat_pred, label)
    print(f"  {label:12s}  {fpr:>8.6f}  {fnr:>8.6f}  "
          f"{tp:>7,}  {fp:>7,}  {fn:>7,}  {tn:>9,}")

print()
print("  FPR: non-entity tokens incorrectly predicted as this label")
print("  FNR: real tokens with this label that the model missed")


# -----------------------------------------------------------------------------
# Error analysis
# -----------------------------------------------------------------------------

print("\n--- Error analysis ---")

test_raw_path = DATA_DIR + 'test_prepared.json'
print(f"Loading raw test entries from {test_raw_path} ...")
with open(test_raw_path) as f:
    test_raw = json.load(f)
print(f"  {len(test_raw):,} entries loaded")

false_positives_per   = []
false_negatives_per   = []
false_positives_email = []
false_negatives_email = []
cross_confusions      = []

for entry, true_seq, pred_seq in zip(test_raw, true_labels, pred_labels):
    tokens = entry['tokens']

    # Token count mismatch can happen due to sub-token alignment edge cases
    if len(tokens) != len(true_seq):
        continue

    for tok, true, pred in zip(tokens, true_seq, pred_seq):
        if true == 'O' and pred in ('B-PER', 'I-PER'):
            false_positives_per.append((entry['sequence'], tok, pred))
        elif true in ('B-PER', 'I-PER') and pred == 'O':
            false_negatives_per.append((entry['sequence'], tok, true))
        elif true == 'O' and pred in ('B-EMAIL', 'I-EMAIL'):
            false_positives_email.append((entry['sequence'], tok, pred))
        elif true in ('B-EMAIL', 'I-EMAIL') and pred == 'O':
            false_negatives_email.append((entry['sequence'], tok, true))

        if (true in ('B-PER', 'I-PER') and pred in ('B-EMAIL', 'I-EMAIL')) or \
           (true in ('B-EMAIL', 'I-EMAIL') and pred in ('B-PER', 'I-PER')):
            cross_confusions.append((entry['sequence'][:100], tok, true, pred))


def show_errors(title, errors, n=5):
    """Print up to n example errors, deduplicated by sentence."""
    print(f"\n{title}  ({len(errors)} total, showing up to {n})")
    if not errors:
        print("  None found.")
        return
    seen, count = set(), 0
    for sentence, token, tag in errors:
        if sentence not in seen:
            print(f"  token    : '{token}'  [{tag}]")
            print(f"  sentence : {sentence[:120]}")
            print()
            seen.add(sentence)
            count += 1
        if count >= n:
            break


show_errors("False positives — PER (predicted name, but wasn't)", false_positives_per)
show_errors("False negatives — PER (missed a real name)",         false_negatives_per)
show_errors("False positives — EMAIL (predicted email, but wasn't)", false_positives_email)
show_errors("False negatives — EMAIL (missed a real email)",         false_negatives_email)

print(f"\nCross-entity confusions (PER predicted as EMAIL or vice versa): "
      f"{len(cross_confusions)}")
for sentence, tok, true, pred in cross_confusions[:5]:
    print(f"  '{tok}': true={true}, pred={pred}")
    print(f"  In: {sentence}")

# -----------------------------------------------------------------------------
# Error summary
# -----------------------------------------------------------------------------

print("\n--- Error count summary ---")
print(f"  False positives PER   : {len(false_positives_per):,}")
print(f"  False negatives PER   : {len(false_negatives_per):,}")
print(f"  False positives EMAIL : {len(false_positives_email):,}")
print(f"  False negatives EMAIL : {len(false_negatives_email):,}")
print(f"  Cross-entity          : {len(cross_confusions):,}")

total_errors = (len(false_positives_per) + len(false_negatives_per) +
                len(false_positives_email) + len(false_negatives_email) +
                len(cross_confusions))
print(f"  Total                 : {total_errors:,} / {total_tokens:,} tokens")

# One pattern worth noting in EMAIL false positives: isolated '.' tokens in
# numeric contexts (e.g. "42.77 %") occasionally get tagged I-EMAIL. This
# seems to be the model over-applying the spaced email pattern it saw during
# training ("alex . morgan @ gmail . com"). Not a critical failure, but worth
# keeping in mind if the model is deployed on text with lots of decimal numbers.
