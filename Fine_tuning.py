"""
Fine-Tuning RoBERTa for PII Detection
=======================================
fine-tunes roberta-base to detect
person names and email addresses, and saves the best checkpoint based on
validation F1.

Training is set up for a T4 GPU (16 GB VRAM):
  - Gradient accumulation brings the effective batch size to 32 without
    running out of memory on the physical 16-sample batches.
  - FP16 and gradient checkpointing are both on to keep memory usage low.
  - Early stopping kicks in after 2 epochs with no improvement, so we're
    not wasting time if the model has already converged.

Hyperparameters at a glance:
  roberta-base | 5 epochs max | lr 2e-5 | effective batch 32 | seqeval F1
"""

import numpy as np
from datasets import load_from_disk
from transformers import (
    RobertaForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    RobertaTokenizerFast,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, classification_report


# -----------------------------------------------------------------------------
# Label definitions
# -----------------------------------------------------------------------------

LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-EMAIL', 'I-EMAIL']
label2id   = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label   = {idx: label for idx, label in enumerate(LABEL_LIST)}

print("Label mapping:")
for label, idx in label2id.items():
    print(f"  {label:12s} -> {idx}")


# -----------------------------------------------------------------------------
# Load tokenized datasets
# -----------------------------------------------------------------------------

DATA_DIR = '/content/drive/MyDrive/Securiti_Internship/Internship_task_data/'

print(f"\nLoading tokenized datasets from {DATA_DIR} ...")

train_tokenized = load_from_disk(DATA_DIR + 'train_tokenized')
val_tokenized   = load_from_disk(DATA_DIR + 'val_tokenized')
test_tokenized  = load_from_disk(DATA_DIR + 'test_tokenized')

print(f"  train_tokenized : {len(train_tokenized):,} examples")
print(f"  val_tokenized   : {len(val_tokenized):,} examples")
print(f"  test_tokenized  : {len(test_tokenized):,} examples")


# -----------------------------------------------------------------------------
# Load pre-trained model
# -----------------------------------------------------------------------------

MODEL_NAME = 'roberta-base'

print(f"\nLoading {MODEL_NAME} with a fresh token classification head...")

model = RobertaForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels = len(LABEL_LIST),
    id2label   = id2label,
    label2id   = label2id,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

# You'll likely see two warnings when this runs:
#   - "unexpected keys: lm_head.*"  -> the original language modeling head,
#     which we don't need for token classification. Safe to ignore.
#   - "missing keys: classifier.*"  -> our new classification head, which
#     starts from scratch and gets trained below. Also expected.


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

tokenizer = RobertaTokenizerFast.from_pretrained(
    MODEL_NAME,
    add_prefix_space=True
)


# -----------------------------------------------------------------------------
# Evaluation metric
# -----------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute seqeval F1 on the validation set after each epoch.

    seqeval treats B-PER + I-PER as a single span — a prediction only
    counts as correct if the entire span boundaries match. This is stricter
    than token-level accuracy and gives a more honest picture of how well
    the model is actually finding named entities.

    Returns overall F1 plus per-entity F1 for PER and EMAIL separately.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Reconstruct string label sequences, ignoring -100 positions
    true_labels = [
        [id2label[l] for l in label_seq if l != -100]
        for label_seq in labels
    ]
    true_preds = [
        [id2label[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    overall_f1 = f1_score(true_labels, true_preds)
    report     = classification_report(true_labels, true_preds, output_dict=True)

    result = {'f1': overall_f1}
    for entity in ['PER', 'EMAIL']:
        if entity in report:
            result[f'{entity}_f1'] = report[entity]['f1-score']

    return result


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------

SAVE_DIR = DATA_DIR + 'roberta_pii_model'

per_device_train_batch_size = 16
gradient_accumulation_steps = 2
num_train_epochs            = 5

# Reserve 10% of training steps for learning rate warmup. This lets the
# optimizer settle before the full learning rate kicks in, which tends to
# improve stability early in fine-tuning.
total_steps  = (len(train_tokenized) // (per_device_train_batch_size * gradient_accumulation_steps)) * num_train_epochs
warmup_steps = int(0.1 * total_steps)
print(f"\nWarmup steps: {warmup_steps}  (10% of {total_steps} total steps)")

training_args = TrainingArguments(

    output_dir = SAVE_DIR,

    # Training duration
    num_train_epochs            = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    per_device_eval_batch_size  = 32,
    gradient_accumulation_steps = gradient_accumulation_steps,   # effective batch = 32

    # Optimizer
    learning_rate = 2e-5,
    weight_decay  = 0.01,
    warmup_steps  = warmup_steps,
    max_grad_norm = 1.0,

    # Checkpointing — save every epoch and keep the best by F1
    eval_strategy          = 'epoch',
    save_strategy          = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model  = 'f1',
    greater_is_better      = True,

    # Logging
    logging_steps = 100,
    report_to     = 'none',

    # Memory optimizations for T4
    fp16                   = True,
    gradient_checkpointing = True,
)

print("\nTraining config summary:")
print(f"  Epochs              : {training_args.num_train_epochs}")
print(f"  Train batch size    : {training_args.per_device_train_batch_size}")
print(f"  Gradient accum      : {training_args.gradient_accumulation_steps}  "
      f"(effective batch = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
print(f"  Eval batch size     : {training_args.per_device_eval_batch_size}")
print(f"  Learning rate       : {training_args.learning_rate}")
print(f"  Weight decay        : {training_args.weight_decay}")
print(f"  Warmup steps        : {training_args.warmup_steps}")
print(f"  Max grad norm       : {training_args.max_grad_norm}")
print(f"  FP16                : {training_args.fp16}")
print(f"  Gradient checkpoint : {training_args.gradient_checkpointing}")
print(f"  Early stopping      : patience = 2 epochs")
print(f"  Saving to           : {SAVE_DIR}")


# -----------------------------------------------------------------------------
# Data collator
# -----------------------------------------------------------------------------

data_collator = DataCollatorForTokenClassification(tokenizer)


# -----------------------------------------------------------------------------
# Trainer setup and training run
# -----------------------------------------------------------------------------

trainer = Trainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_tokenized,
    eval_dataset     = val_tokenized,
    processing_class = tokenizer,       # renamed from 'tokenizer' in newer HF versions
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
    callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n" + "-" * 50)
print("Starting training...")
print("-" * 50 + "\n")

train_result = trainer.train()

print("\n" + "-" * 50)
print("Training finished.")
print("-" * 50)
print(f"  Steps completed : {train_result.global_step}")
print(f"  Training loss   : {train_result.training_loss:.4f}  (averaged across all epochs)")

# Print per-epoch validation results for a quick overview
print("\nValidation results per epoch:")
for entry in trainer.state.log_history:
    if 'eval_f1' in entry:
        print(f"  Epoch {entry.get('epoch', '?'):.0f}:  "
              f"loss = {entry.get('eval_loss', float('nan')):.4f},  "
              f"F1 = {entry['eval_f1']:.4f}")


# -----------------------------------------------------------------------------
# Save the best checkpoint
# -----------------------------------------------------------------------------

FINAL_MODEL_DIR = DATA_DIR + 'roberta_pii_final'

print(f"\nSaving best model to {FINAL_MODEL_DIR} ...")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# When saving/loading you may see a LayerNorm naming warning (beta/gamma vs
# weight/bias). This is a PyTorch vs TensorFlow naming convention difference —
# the actual weights are identical, so the model behaves correctly either way.
