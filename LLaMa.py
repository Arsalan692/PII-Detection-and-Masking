"""
Zero-Shot PII Detection with LLaMA 1B
=======================================
Runs Llama-3.2-1B-Instruct in zero-shot mode on a stratified 325-example
subset of the test data and evaluates how well it identifies person names
and email addresses without any task-specific training.

Results are compared against the fine-tuned RoBERTa model from Step 4.

Hardware note: uses float16 throughout — T4 GPUs don't handle bfloat16 well,
so that's intentionally avoided here even though newer GPUs prefer it.
"""

import json
import re
import random
import numpy as np
from sklearn.metrics import confusion_matrix

random.seed(42)


# -----------------------------------------------------------------------------
# Authentication and model loading
# -----------------------------------------------------------------------------

import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGING_FACE_TOKEN"))

import torch
from transformers import pipeline

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading {MODEL_ID} in float16 ...")
llama_pipeline = pipeline(
    "text-generation",
    model      = MODEL_ID,
    dtype      = torch.float16,
    device_map = "auto",
)
print("Model ready.")


# -----------------------------------------------------------------------------
# Load test data
# -----------------------------------------------------------------------------

DATA_DIR = '/content/drive/MyDrive/Securiti_Internship/Internship_task_data/'

with open(DATA_DIR + 'test_prepared.json', 'r') as f:
    test_data = json.load(f)
print(f"Test entries loaded: {len(test_data):,}")


# -----------------------------------------------------------------------------
# Stratified subset selection
# -----------------------------------------------------------------------------

def select_stratified_subset(data, n_with_name=250, n_with_email=75):
    """Sample a balanced evaluation subset from the test data.

    Pulls 250 entries that have a person name but no email, and 75 that have
    an email (which may also have a name). Keeping these buckets separate
    avoids double-counting when computing per-entity metrics.
    """
    with_email = [e for e in data if 'B-EMAIL' in e['ner_tags']]
    with_name  = [e for e in data if 'B-PER'   in e['ner_tags']
                                  and 'B-EMAIL' not in e['ner_tags']]

    sampled_email = random.sample(with_email, min(n_with_email, len(with_email)))
    sampled_name  = random.sample(with_name,  min(n_with_name,  len(with_name)))

    subset = sampled_email + sampled_name
    random.shuffle(subset)

    print(f"\nEvaluation subset: {len(subset)} examples total")
    print(f"  Name only (no email) : {len(sampled_name)}")
    print(f"  With email           : {len(sampled_email)}")
    return subset

eval_subset = select_stratified_subset(test_data)


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def build_prompt(sentence):
    """Build a zero-shot chat prompt for the LLaMA instruct model.

    No few-shot examples are included — the whole point is to see how the
    model performs without any task-specific demonstrations.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a PII extraction assistant. "
                "Extract person names and email addresses from the given text. "
                "Respond with ONLY a JSON object in this exact format: "
                '{"names": ["full name1", "full name2"], "emails": ["email1", "email2"]} '
                "If nothing is found, return empty lists. "
                "Do NOT include any explanation or extra text."
            )
        },
        {
            "role": "user",
            "content": f'Text: "{sentence}"\n\nExtract all person names and email addresses:'
        }
    ]
    return messages


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------

def parse_llama_response(raw_text):
    """Extract names and emails from the model's raw output.

    First tries to find and parse a JSON object anywhere in the response.
    If that fails (malformed JSON, extra prose, etc.), falls back to a regex
    scan for anything that looks like an email address.
    """
    json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            names  = parsed.get('names',  [])
            emails = parsed.get('emails', [])
            if not isinstance(names,  list): names  = []
            if not isinstance(emails, list): emails = []
            return {'names': names, 'emails': emails}
        except json.JSONDecodeError:
            pass

    # JSON parse failed — at least try to recover emails via regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return {'names': [], 'emails': re.findall(email_pattern, raw_text)}


# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------

def run_llama_evaluation(subset):
    """Run the model over the full subset and collect structured results.

    For each entry, reconstructs the ground-truth entity strings from the
    token/tag sequences, runs the model, parses the response, and stores
    everything needed for downstream metric calculation.
    """
    results = []
    print("\nRunning inference ...")

    for i, entry in enumerate(subset):
        sentence = entry['sequence']
        ner_tags = entry['ner_tags']
        tokens   = entry['tokens']

        has_name  = 'B-PER'   in ner_tags
        has_email = 'B-EMAIL' in ner_tags

        # Reconstruct full entity strings from the BIO tag sequence
        true_names, true_emails = [], []
        j = 0
        while j < len(tokens):
            if ner_tags[j] == 'B-PER':
                name = tokens[j]; j += 1
                while j < len(tokens) and ner_tags[j] == 'I-PER':
                    name += ' ' + tokens[j]; j += 1
                true_names.append(name)
            elif ner_tags[j] == 'B-EMAIL':
                parts = [tokens[j]]; j += 1
                while j < len(tokens) and ner_tags[j] == 'I-EMAIL':
                    parts.append(tokens[j]); j += 1
                true_emails.append(''.join(parts))
            else:
                j += 1

        try:
            output = llama_pipeline(
                build_prompt(sentence),
                max_new_tokens = 150,
                do_sample      = False,
                pad_token_id   = llama_pipeline.tokenizer.eos_token_id,
            )
            raw_output = output[0]['generated_text'][-1]['content'].strip()
        except Exception as e:
            raw_output = '{}'
            print(f"  Warning: inference failed on example {i}: {e}")

        prediction = parse_llama_response(raw_output)

        results.append({
            'sentence'      : sentence,
            'true_names'    : true_names,
            'true_emails'   : true_emails,
            'has_name'      : has_name,
            'has_email'     : has_email,
            'pred_names'    : prediction['names'],
            'pred_emails'   : prediction['emails'],
            'pred_has_name' : len(prediction['names'])  > 0,
            'pred_has_email': len(prediction['emails']) > 0,
            'raw_output'    : raw_output,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(subset)} done")

    print(f"\nInference complete — {len(results)} examples processed")
    return results

results = run_llama_evaluation(eval_subset)


# -----------------------------------------------------------------------------
# Sample outputs
# -----------------------------------------------------------------------------

print("\n--- First 5 raw outputs ---")
for r in results[:5]:
    print(f"  sentence     : {r['sentence'][:80]}")
    print(f"  true names   : {r['true_names']}")
    print(f"  true emails  : {r['true_emails']}")
    print(f"  raw output   : {r['raw_output'][:120]}")
    print(f"  parsed names : {r['pred_names']}")
    print(f"  parsed emails: {r['pred_emails']}")
    print()


# -----------------------------------------------------------------------------
# Binary classification metrics (presence/absence per sentence)
# -----------------------------------------------------------------------------

def calculate_metrics(results, entity_type):
    """Compute precision, recall, F1, accuracy, FPR, and FNR for one entity type.

    Treats each sentence as a binary classification: does it contain this
    entity type or not? This gives a high-level picture of how reliably the
    model detects the presence of PII, independent of exact string matching.
    """
    if entity_type == 'name':
        true = [1 if r['has_name']       else 0 for r in results]
        pred = [1 if r['pred_has_name']  else 0 for r in results]
    else:
        true = [1 if r['has_email']      else 0 for r in results]
        pred = [1 if r['pred_has_email'] else 0 for r in results]

    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1':        round(f1,        4), 'accuracy': round(accuracy, 4),
        'fpr':       round(fpr,       4), 'fnr':      round(fnr,      4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }

name_metrics  = calculate_metrics(results, 'name')
email_metrics = calculate_metrics(results, 'email')

print("\n--- LLaMA zero-shot results (sentence-level) ---\n")
print(f"  {'Metric':12s}  {'PER':>10}  {'EMAIL':>10}")
print(f"  {'-' * 35}")
for metric in ['precision', 'recall', 'f1', 'accuracy', 'fpr', 'fnr']:
    print(f"  {metric:12s}  {name_metrics[metric]:>10.4f}  {email_metrics[metric]:>10.4f}")

print(f"\n  Confusion matrix breakdown:")
print(f"  {'':12s}  {'PER':>10}  {'EMAIL':>10}")
print(f"  {'-' * 35}")
for key in ['tp', 'tn', 'fp', 'fn']:
    print(f"  {key:12s}  {name_metrics[key]:>10}  {email_metrics[key]:>10}")


# -----------------------------------------------------------------------------
# Entity-level evaluation (exact/substring string matching)
# -----------------------------------------------------------------------------

def entity_level_eval(results):
    """Check whether the specific entity strings were extracted correctly.

    PER uses a case-insensitive substring match — either the prediction is
    contained in the ground truth or vice versa. EMAIL uses exact match after
    lowercasing and stripping spaces (to handle spaced email formats).
    """
    per_tp = per_fp = per_fn = 0
    email_tp = email_fp = email_fn = 0

    for r in results:
        for true_name in r['true_names']:
            tl = true_name.lower()
            if any(tl in p.lower() or p.lower() in tl for p in r['pred_names']):
                per_tp += 1
            else:
                per_fn += 1
        for pred_name in r['pred_names']:
            pl = pred_name.lower()
            if not any(pl in t.lower() or t.lower() in pl for t in r['true_names']):
                per_fp += 1

        for true_email in r['true_emails']:
            tn = true_email.lower().replace(' ', '')
            if any(tn == p.lower().replace(' ', '') for p in r['pred_emails']):
                email_tp += 1
            else:
                email_fn += 1
        for pred_email in r['pred_emails']:
            pn = pred_email.lower().replace(' ', '')
            if not any(pn == t.lower().replace(' ', '') for t in r['true_emails']):
                email_fp += 1

    return per_tp, per_fp, per_fn, email_tp, email_fp, email_fn

per_tp, per_fp, per_fn, email_tp, email_fp, email_fn = entity_level_eval(results)

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

per_precision,   per_recall,   per_f1   = _prf(per_tp,   per_fp,   per_fn)
email_precision, email_recall, email_f1 = _prf(email_tp, email_fp, email_fn)

print("\n--- Entity-level evaluation (string matching) ---\n")
print(f"  {'Metric':12s}  {'PER':>10}  {'EMAIL':>10}")
print(f"  {'-' * 35}")
print(f"  {'precision':12s}  {per_precision:>10.4f}  {email_precision:>10.4f}")
print(f"  {'recall':12s}  {per_recall:>10.4f}  {email_recall:>10.4f}")
print(f"  {'f1':12s}  {per_f1:>10.4f}  {email_f1:>10.4f}")

print(f"\n  Entity counts:")
print(f"  {'':12s}  {'PER':>10}  {'EMAIL':>10}")
print(f"  {'-' * 35}")
for label, pv, ev in [('tp', per_tp, email_tp), ('fp', per_fp, email_fp), ('fn', per_fn, email_fn)]:
    print(f"  {label:12s}  {pv:>10}  {ev:>10}")


# -----------------------------------------------------------------------------
# Failure mode analysis
# -----------------------------------------------------------------------------

print("\n--- Failure mode analysis ---")

bad_json       = []
hallucinations = []
partial_names  = []
format_issues  = []

for r in results:
    raw = r['raw_output']

    if not re.search(r'\{[^{}]*\}', raw):
        bad_json.append(r)

    if not r['has_name']  and r['pred_has_name']:
        hallucinations.append(('name',  r))
    if not r['has_email'] and r['pred_has_email']:
        hallucinations.append(('email', r))

    for true_name in r['true_names']:
        for pred_name in r['pred_names']:
            if pred_name in true_name and pred_name != true_name:
                partial_names.append((true_name, pred_name, r['sentence']))

    if re.search(r'\{[^{}]*\}', raw):
        stripped = raw.strip()
        if not (stripped.startswith('{') and stripped.endswith('}')):
            format_issues.append(r)

print(f"\n  Bad JSON responses   : {len(bad_json)}")
print(f"  Hallucinations       : {len(hallucinations)}")
print(f"  Partial name matches : {len(partial_names)}")
print(f"  Format issues        : {len(format_issues)}")

if bad_json:
    print(f"\nBad JSON examples (up to 3):")
    for r in bad_json[:3]:
        print(f"  sentence   : {r['sentence'][:80]}")
        print(f"  raw output : {r['raw_output'][:100]}")
        print()

if hallucinations:
    print(f"Hallucination examples (up to 3):")
    for entity_type, r in hallucinations[:3]:
        print(f"  type      : {entity_type}")
        print(f"  sentence  : {r['sentence'][:80]}")
        print(f"  predicted : {r['pred_names'] if entity_type == 'name' else r['pred_emails']}")
        print()

if partial_names:
    print(f"Partial name examples (up to 3):")
    for true_name, pred_name, sentence in partial_names[:3]:
        print(f"  true: '{true_name}'  ->  predicted: '{pred_name}'")
        print(f"  in  : {sentence[:80]}")
        print()


# -----------------------------------------------------------------------------
# Save results to disk
# -----------------------------------------------------------------------------

print("\nSaving results ...")

llama_metrics = {
    'name_metrics' : name_metrics,
    'email_metrics': email_metrics,
    'entity_level' : {
        'per'  : {'tp': per_tp,   'fp': per_fp,   'fn': per_fn,
                  'precision': per_precision,   'recall': per_recall,   'f1': per_f1},
        'email': {'tp': email_tp, 'fp': email_fp, 'fn': email_fn,
                  'precision': email_precision, 'recall': email_recall, 'f1': email_f1},
    },
    'n_examples': len(results),
    'challenges': {
        'bad_json'      : len(bad_json),
        'hallucinations': len(hallucinations),
        'partial_names' : len(partial_names),
        'format_issues' : len(format_issues),
    },
}

with open(DATA_DIR + 'llama_metrics.json', 'w') as f:
    json.dump(llama_metrics, f, indent=2)

results_to_save = [{k: v for k, v in r.items() if k != 'raw_output'} for r in results]
with open(DATA_DIR + 'llama_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=2)

print("  llama_metrics.json and llama_results.json written.")