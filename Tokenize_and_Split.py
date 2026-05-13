"""
Tokenization and Train/Validation Split
========================================
Loads the prepared NER data, carves out a validation split, tokenizes
everything for RoBERTa, and writes three ready-to-use dataset folders.

A few things to be aware of:
  - The test set is never touched during training or validation — it stays
    completely held out until final evaluation.
  - The train/val split is stratified on whether an entry has an email,
    since emails only appear in ~15% of entries and a naive random split
    could skew the distribution.
  - RoBERTa sub-tokenizes emails (e.g. "alex.morgan@gmail.com" becomes
    ["alex", ".", "morgan", "@", "gmail", ".", "com"]). The first sub-token
    gets B-EMAIL and the rest get I-EMAIL rather than -100, so the model
    actually learns to predict I-EMAIL continuations during training.
"""

import json
import random
from datasets import Dataset
from transformers import RobertaTokenizerFast

random.seed(42)


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
# Tokenizer
# -----------------------------------------------------------------------------

MODEL_NAME = 'roberta-base'
tokenizer  = RobertaTokenizerFast.from_pretrained(
    MODEL_NAME,
    add_prefix_space=True
)
print(f"\nLoaded tokenizer: {MODEL_NAME}")


# -----------------------------------------------------------------------------
# Stratified train/validation split
# -----------------------------------------------------------------------------

def stratified_split(dataset, val_ratio=0.10):
    """Split the dataset into train and validation while preserving the
    proportion of email-containing entries in each subset.

    A plain random split risks under-representing emails in validation
    since they only appear in ~15% of entries. Stratifying on that flag
    keeps the ratio consistent across both splits.

    The approach is straightforward — separate into two buckets (has email /
    no email), sample val_ratio from each independently, then recombine
    and shuffle.

    Example with 100 entries (15 with email, 85 without):
        email bucket    -> 2 to val,  13 to train
        no-email bucket -> 9 to val,  76 to train
        val total       -> 11 entries, still ~15% have emails
        train total     -> 89 entries, still ~15% have emails
    """
    has_email = [e for e in dataset if 'B-EMAIL' in e['ner_tags']]
    no_email  = [e for e in dataset if 'B-EMAIL' not in e['ner_tags']]

    random.shuffle(has_email)
    random.shuffle(no_email)

    n_val_email    = int(len(has_email) * val_ratio)
    n_val_no_email = int(len(no_email)  * val_ratio)

    val_data   = has_email[:n_val_email]  + no_email[:n_val_no_email]
    train_data = has_email[n_val_email:]  + no_email[n_val_no_email:]

    # Shuffle so email entries aren't all clumped at the top
    random.shuffle(val_data)
    random.shuffle(train_data)

    return train_data, val_data


# -----------------------------------------------------------------------------
# Tokenization with label alignment
# -----------------------------------------------------------------------------

def tokenize_and_align_labels(batch):
    """Tokenize a batch of word-level sequences and align NER labels to
    RoBERTa's sub-token output.

    The alignment rules are:
      - Special tokens (CLS, SEP, PAD) get -100 and are ignored by the loss.
      - The first sub-token of each word gets the word's real label.
      - Subsequent sub-tokens of the same word get -100, except when the
        word is tagged B-EMAIL — in that case they get I-EMAIL.

    The B-EMAIL exception is important. RoBERTa splits something like
    "alex.morgan@gmail.com" into seven pieces. Without this rule, the model
    would never see I-EMAIL during training and couldn't predict it at
    inference time. Propagating I-EMAIL to those continuation sub-tokens
    teaches the model that everything after the first piece of an email
    address is still part of that address.
    """
    tokenized = tokenizer(
        batch['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=256,
    )

    all_labels = []
    for i, ner_tags in enumerate(batch['ner_tags']):
        word_ids         = tokenized.word_ids(batch_index=i)
        aligned_labels   = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First sub-token of this word — assign its real label
                aligned_labels.append(label2id[ner_tags[word_id]])
            else:
                # Continuation sub-token — propagate I-EMAIL for email words,
                # ignore everything else
                if ner_tags[word_id] == 'B-EMAIL':
                    aligned_labels.append(label2id['I-EMAIL'])
                else:
                    aligned_labels.append(-100)
            previous_word_id = word_id

        all_labels.append(aligned_labels)

    tokenized['labels'] = all_labels
    return tokenized


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def to_hf_dataset(data):
    """Reformat a list of entry dicts into a HuggingFace Dataset object.

    Input:  [{'tokens': [...], 'ner_tags': [...]}, ...]
    Output: Dataset with 'tokens' and 'ner_tags' columns
    """
    return Dataset.from_dict({
        'tokens':   [d['tokens']   for d in data],
        'ner_tags': [d['ner_tags'] for d in data],
    })


def count_tags(data):
    """Tally how often each NER tag appears across a list of entries."""
    counts = {}
    for entry in data:
        for tag in entry['ner_tags']:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    DATA_DIR = '/content/drive/MyDrive/Securiti_Internship/Internship_task_data/'

    # Load the prepared files from Step 1
    print(f"\nLoading prepared data from {DATA_DIR} ...")
    with open(DATA_DIR + 'train_prepared.json', 'r') as f:
        train_prepared = json.load(f)
    with open(DATA_DIR + 'test_prepared.json', 'r') as f:
        test_prepared = json.load(f)

    print(f"  train_prepared.json : {len(train_prepared):,} entries")
    print(f"  test_prepared.json  : {len(test_prepared):,} entries")

    # -------------------------------------------------------------------------
    # Train / validation split
    # -------------------------------------------------------------------------
    print("\nSplitting training data (90% train, 10% validation, stratified on email)...")
    train_data, val_data = stratified_split(train_prepared, val_ratio=0.10)

    print(f"  Train      : {len(train_data):,} entries")
    print(f"  Validation : {len(val_data):,} entries")
    print(f"  Test       : {len(test_prepared):,} entries (held out)")

    # Verify that email proportion is consistent across all three splits
    print("\n--- Email distribution check ---")
    for name, data in [('Train', train_data), ('Validation', val_data), ('Test', test_prepared)]:
        tags        = count_tags(data)
        total       = len(data)
        email_count = sum(1 for e in data if 'B-EMAIL' in e['ner_tags'])
        print(f"  {name:12s} : {total:,} entries | "
              f"with email: {email_count:,} ({email_count / total * 100:.1f}%) | "
              f"B-PER: {tags.get('B-PER', 0):,} | "
              f"B-EMAIL: {tags.get('B-EMAIL', 0):,}")

    # -------------------------------------------------------------------------
    # Check how many sequences will get truncated
    # -------------------------------------------------------------------------
    all_entries = train_data + val_data + test_prepared
    long_seqs = sum(
        1 for entry in all_entries
        if len(tokenizer(entry['tokens'], is_split_into_words=True,
                         truncation=False)['input_ids']) > 256
    )
    print(f"\n  Sequences over 256 sub-tokens: {long_seqs} "
          f"({long_seqs / len(all_entries) * 100:.1f}%)")
    if long_seqs > 0:
        print(f"  Note: those {long_seqs} sequences will be truncated during tokenization.")

    # -------------------------------------------------------------------------
    # Convert to HuggingFace Dataset and tokenize
    # -------------------------------------------------------------------------
    print("\nConverting to HuggingFace Dataset format...")
    train_dataset = to_hf_dataset(train_data)
    val_dataset   = to_hf_dataset(val_data)
    test_dataset  = to_hf_dataset(test_prepared)

    print("Tokenizing all three splits...")
    train_tokenized = train_dataset.map(
        tokenize_and_align_labels, batched=True, batch_size=256,
        remove_columns=['tokens', 'ner_tags']
    )
    val_tokenized = val_dataset.map(
        tokenize_and_align_labels, batched=True, batch_size=256,
        remove_columns=['tokens', 'ner_tags']
    )
    test_tokenized = test_dataset.map(
        tokenize_and_align_labels, batched=True, batch_size=256,
        remove_columns=['tokens', 'ner_tags']
    )

    print(f"  train_tokenized : {len(train_tokenized):,} examples")
    print(f"  val_tokenized   : {len(val_tokenized):,} examples")
    print(f"  test_tokenized  : {len(test_tokenized):,} examples")
    print(f"  Columns         : {train_tokenized.column_names}")

    # Quick attention mask sanity check on one sample
    sample    = train_tokenized[0]
    real_toks = sum(sample['attention_mask'])
    pad_toks  = len(sample['attention_mask']) - real_toks
    print(f"\n  Sample attention mask: {real_toks} real tokens, {pad_toks} padding")

    # -------------------------------------------------------------------------
    # Sub-token alignment check — find a B-EMAIL entry and print the mapping
    # -------------------------------------------------------------------------
    print("\n--- Sub-token label alignment (B-EMAIL example) ---")

    email_sample_idx = next(
        (idx for idx, e in enumerate(train_data) if 'B-EMAIL' in e['ner_tags']),
        None
    )

    # Fall back to the first entry if no email example exists in the batch
    display_idx = email_sample_idx if email_sample_idx is not None else 0
    if email_sample_idx is None:
        print("  No B-EMAIL entry found — showing first entry instead.")

    sample_original  = train_data[display_idx]
    sample_tokenized = train_tokenized[display_idx]

    print(f"\n  Sentence: {sample_original.get('sequence', ' '.join(sample_original['tokens']))}")
    print(f"\n  {'Sub-token':<22} {'label_id':>8}  label_name")
    print(f"  {'-' * 45}")

    subtokens = tokenizer.convert_ids_to_tokens(sample_tokenized['input_ids'])
    for subtoken, input_id, label in zip(
        subtokens,
        sample_tokenized['input_ids'],
        sample_tokenized['labels']
    ):
        if input_id == tokenizer.pad_token_id:
            break
        label_name = id2label[label] if label != -100 else '-100 (ignored)'
        print(f"  {subtoken:<22} {label:>8}  {label_name}")

    # -------------------------------------------------------------------------
    # Confirm I-EMAIL labels are actually present in the tokenized training set
    # -------------------------------------------------------------------------
    print("\n--- I-EMAIL label count in tokenized training data ---")
    i_email_count = sum(
        1 for example in train_tokenized
        for label_id in example['labels']
        if label_id == label2id['I-EMAIL']
    )
    print(f"  I-EMAIL labels found: {i_email_count:,}")
    if i_email_count > 0:
        print("  Sub-token propagation is working correctly.")
    else:
        print("  None found — double-check the label alignment logic above.")

    # -------------------------------------------------------------------------
    # Save to disk
    # -------------------------------------------------------------------------
    print(f"\nSaving tokenized datasets to {DATA_DIR} ...")
    train_tokenized.save_to_disk(DATA_DIR + 'train_tokenized')
    val_tokenized.save_to_disk(DATA_DIR + 'val_tokenized')
    test_tokenized.save_to_disk(DATA_DIR + 'test_tokenized')

    print("\nDone. Three dataset folders written:")
    print(f"  train_tokenized/  <- model trains on this")
    print(f"  val_tokenized/    <- used to monitor training progress")
    print(f"  test_tokenized/   <- final evaluation only, never seen during training")