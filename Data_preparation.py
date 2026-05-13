"""
Data Preparation: Synthetic Email Injection
============================================
This script augments the NER dataset by generating realistic email addresses
from person names already present in the data and appending short contact
sentences to a subset of entries.

A few things worth noting about how this is set up:
  - Train and test splits use completely separate sentence templates to avoid
    data leakage. The model shouldn't just be memorizing sentence patterns.
  - Roughly 30% of injected emails are written in spaced format
    (e.g. "alex . morgan @ gmail . com") so that the email spans multiple
    whitespace-delimited tokens, giving us both B-EMAIL and I-EMAIL labels.
  - The other ~70% are standard single-token emails, which only produce B-EMAIL.
  - Only 15% of entries that already contain a person name get an email injected.
"""

import json
import random

random.seed(42)


# -----------------------------------------------------------------------------
# Email address components
# -----------------------------------------------------------------------------

DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "protonmail.com", "mail.com", "aol.com",
    "nfl.com", "nba.com", "wikipedia.org", "nyt.com",
    "bbc.co.uk", "cnn.com", "university.edu", "company.org",
]
# Domains like bbc.co.uk have two dots, so spaced versions of those
# will produce a few extra I-EMAIL tokens. That's fine and expected.

EMAIL_PATTERNS = [
    "{first}.{last}@{domain}",
    "{f_init}{last}@{domain}",
    "{first}_{last}@{domain}",
    "{first}{l_init}@{domain}",
    "{first}{num}@{domain}",
]


def make_email(first_name, last_name):
    """Build a plausible email address from a person's first and last name.

    If only a single name token is available (no last name), we fall back
    to patterns that only use the first name to avoid things like
    'alex.alex@gmail.com'.

    Returns a plain string, e.g. 'alex.morgan@gmail.com'.
    """
    first = first_name.lower().strip('.,"\'; ')
    last  = last_name.lower().strip('.,"\'; ') if last_name else ''
    domain = random.choice(DOMAINS)

    if last:
        patterns = [
            "{first}.{last}@{domain}",
            "{f_init}{last}@{domain}",
            "{first}_{last}@{domain}",
            "{first}{l_init}@{domain}",
            "{first}{num}@{domain}",
        ]
    else:
        patterns = [
            "{first}@{domain}",
            "{first}{num}@{domain}",
            "{f_init}{first}@{domain}",
        ]

    pattern = random.choice(patterns)
    return pattern.format(
        first=first,
        last=last,
        f_init=first[0] if first else 'x',
        l_init=last[0] if last else 'y',
        num=random.randint(1, 99),
        domain=domain,
    )


def make_spaced_email(first_name, last_name):
    """Generate an email and also return a whitespace-padded version of it.

    The spaced version puts spaces around '.' and '@', so when the sentence
    is tokenized on whitespace, each part of the email becomes its own token.
    This is what gives us B-EMAIL on the first token and I-EMAIL on the rest.

    Example:
        normal  -> 'alex.morgan@gmail.com'
        spaced  -> 'alex . morgan @ gmail . com'

    Returns:
        original (str) - the regular email
        spaced   (str) - the space-padded version for sentence injection
    """
    original = make_email(first_name, last_name)
    spaced = original.replace('.', ' . ').replace('@', ' @ ')
    while '  ' in spaced:
        spaced = spaced.replace('  ', ' ')
    return original, spaced.strip()


# -----------------------------------------------------------------------------
# Sentence templates for email injection
# -----------------------------------------------------------------------------

# These are the templates used when building training examples. The model
# will see these patterns during training, so we keep them separate from
# the test templates below.
TRAIN_TEMPLATES = [
    "For more information , contact {email} .",
    "Inquiries can be sent to {email} .",
    "The official contact email is {email} .",
    "Press inquiries should be directed to {email} .",
    "The official press contact is {email} .",
    "Media requests can be sent to {email} .",
    "Correspondence can be sent to {email} .",
    "The address {email} is listed on the official website .",
    "Further details are available from {email} .",
    "The contact email listed is {email} .",
    # spaced format — these produce multi-token B-EMAIL + I-EMAIL sequences
    "Reach out at {spaced_email} for details .",
    "Please forward questions to {spaced_email} .",
    "The listed email address was {spaced_email} .",
    "Send feedback to {spaced_email} anytime .",
    "All correspondence goes to {spaced_email} .",
]

# Completely different templates for test data. The model has never seen
# these phrasings, so this checks whether it can generalize rather than
# just pattern-match on familiar sentences.
TEST_TEMPLATES = [
    "The email provided was {email} .",
    "A message was sent from {email} .",
    "The inbox associated is {email} .",
    "An email was received from {email} .",
    "The registered email is {email} .",
    "The account is linked to {email} .",
    "A reply was sent to {email} .",
    "The record shows {email} as the contact .",
    "Communications were exchanged via {email} .",
    "The email on file reads {email} .",
    # spaced format for test set
    "A notification was dispatched to {spaced_email} .",
    "The system logged {spaced_email} as the sender .",
    "Documents were forwarded to {spaced_email} .",
    "The contact detail on record is {spaced_email} .",
    "An automated response was sent to {spaced_email} .",
]

# Template indices that use {spaced_email} — needed so injection logic
# knows which formatting path to take.
_SPACED_TRAIN_IDXS = {10, 11, 12, 13, 14}
_SPACED_TEST_IDXS  = {10, 11, 12, 13, 14}


# -----------------------------------------------------------------------------
# Span detection
# -----------------------------------------------------------------------------

def find_person_spans(ner_tags):
    """Scan a tag sequence and return all B-PER/I-PER spans as (start, end) pairs.

    Both indices are inclusive. For example:
        tags    = ['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'O']
        returns -> [(1, 2), (4, 4)]
    """
    spans = []
    i = 0
    while i < len(ner_tags):
        if ner_tags[i] == 'B-PER':
            start = end = i
            while end + 1 < len(ner_tags) and ner_tags[end + 1] == 'I-PER':
                end += 1
            spans.append((start, end))
            i = end + 1
        else:
            i += 1
    return spans


# -----------------------------------------------------------------------------
# Single-entry injection
# -----------------------------------------------------------------------------

def inject_email_into_entry(entry, templates, spaced_idxs):
    """Append a contact sentence with an email address to a single data entry.

    Picks a random person span from the entry to derive the email username,
    then selects a random template and builds the new sentence. The resulting
    tokens and tags are appended to the existing ones.

    For normal (non-spaced) emails, the whole address is one token -> B-EMAIL.
    For spaced emails, each part becomes its own token -> B-EMAIL, then I-EMAIL
    for every subsequent token belonging to that address.
    """
    tokens   = entry['tokens'][:]
    ner_tags = entry['ner_tags'][:]

    spans = find_person_spans(ner_tags)
    if not spans:
        return entry

    start, end = random.choice(spans)
    first_name = tokens[start]
    # If the span is a single token there's no last name to work with
    last_name  = tokens[end] if end > start else ''

    tidx     = random.randrange(len(templates))
    template = templates[tidx]

    if tidx in spaced_idxs:
        # Spaced path: email splits across multiple tokens
        original, spaced = make_spaced_email(first_name, last_name)
        filled       = template.replace('{spaced_email}', spaced)
        new_tokens   = filled.split()
        email_tokens = spaced.split()

        # Locate where the email starts by position rather than value matching,
        # since tokens like '.' or '@' could appear elsewhere in the sentence.
        email_start = None
        for k in range(len(new_tokens) - len(email_tokens) + 1):
            if new_tokens[k:k + len(email_tokens)] == email_tokens:
                email_start = k
                break

        new_tags = []
        if email_start is not None:
            for k in range(len(new_tokens)):
                if k == email_start:
                    new_tags.append('B-EMAIL')
                elif email_start < k < email_start + len(email_tokens):
                    new_tags.append('I-EMAIL')
                else:
                    new_tags.append('O')
        else:
            # Positional scan failed — fall back to value-based matching
            email_idx = 0
            for tok in new_tokens:
                if email_idx < len(email_tokens) and tok == email_tokens[email_idx]:
                    new_tags.append('B-EMAIL' if email_idx == 0 else 'I-EMAIL')
                    email_idx += 1
                else:
                    new_tags.append('O')
    else:
        # Standard path: email is a single whitespace token -> just B-EMAIL
        email      = make_email(first_name, last_name)
        filled     = template.replace('{email}', email)
        new_tokens = filled.split()
        new_tags   = ['B-EMAIL' if tok == email else 'O' for tok in new_tokens]

    return {
        'lang':     entry['lang'],
        'tokens':   tokens + new_tokens,
        'ner_tags': ner_tags + new_tags,
        'sequence': ' '.join(tokens + new_tokens),
    }


# -----------------------------------------------------------------------------
# Dataset-level injection
# -----------------------------------------------------------------------------

def inject_emails(dataset, templates, spaced_idxs, injection_rate=0.15):
    """Inject emails into a random subset of entries that contain person names.

    Only entries with at least one B-PER tag are eligible. We inject into
    roughly `injection_rate` fraction of those. Train and test each receive
    their own template set so sentence patterns don't overlap between splits.
    """
    per_entries    = [i for i, e in enumerate(dataset) if 'B-PER' in e['ner_tags']]
    n_to_inject    = int(len(per_entries) * injection_rate)
    chosen_indices = set(random.sample(per_entries, n_to_inject))

    result = []
    count  = 0
    for i, entry in enumerate(dataset):
        if i in chosen_indices and 'B-PER' in entry['ner_tags']:
            result.append(inject_email_into_entry(entry, templates, spaced_idxs))
            count += 1
        else:
            result.append(entry)

    print(f"  Injected into {count:,} of {len(per_entries):,} eligible entries "
          f"({count / len(dataset) * 100:.1f}% of the full {len(dataset):,})")
    return result


# -----------------------------------------------------------------------------
# Tag counting utility
# -----------------------------------------------------------------------------

def count_tags(dataset):
    """Count how many times each NER tag appears across the entire dataset."""
    counts = {}
    for entry in dataset:
        for tag in entry['ner_tags']:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    DATA_DIR = '/content/drive/MyDrive/Securiti_Internship/Internship_task_data/'

    print("Reading source data files...")
    with open(DATA_DIR + 'data.json', 'r') as f:
        train_data = json.load(f)
    with open(DATA_DIR + 'test_data.json', 'r') as f:
        test_data = json.load(f)

    print(f"  Training set : {len(train_data):,} entries")
    print(f"  Test set     : {len(test_data):,} entries")

    print("\nRunning email injection on training data...")
    train_prepared = inject_emails(
        train_data, TRAIN_TEMPLATES, _SPACED_TRAIN_IDXS, injection_rate=0.15
    )

    print("Running email injection on test data...")
    test_prepared = inject_emails(
        test_data, TEST_TEMPLATES, _SPACED_TEST_IDXS, injection_rate=0.15
    )

    # -------------------------------------------------------------------------
    # Tag distribution after injection
    # -------------------------------------------------------------------------
    print("\n--- Tag counts after injection ---")

    print("\nTraining data:")
    train_tag_counts = count_tags(train_prepared)
    for tag, count in sorted(train_tag_counts.items()):
        print(f"  {tag:12s} : {count:,}")

    print("\nTest data:")
    test_tag_counts = count_tags(test_prepared)
    for tag, count in sorted(test_tag_counts.items()):
        print(f"  {tag:12s} : {count:,}")

    # Quick sanity check on B-EMAIL vs I-EMAIL ratio. Since ~30% of injected
    # emails use the spaced format, we expect a non-trivial I-EMAIL count.
    # A very low ratio usually means the spaced injection isn't firing correctly.
    for label, tc in [("TRAIN", train_tag_counts), ("TEST", test_tag_counts)]:
        b_email = tc.get('B-EMAIL', 0)
        i_email = tc.get('I-EMAIL', 0)
        if b_email == 0:
            print(f"\n  WARNING [{label}]: no B-EMAIL tags found — check injection logic.")
        else:
            ratio = i_email / b_email
            print(f"\n  [{label}] I-EMAIL / B-EMAIL ratio: {ratio:.2f}  "
                  f"(B-EMAIL={b_email:,}, I-EMAIL={i_email:,})")
            if ratio < 0.1:
                print(f"  NOTE [{label}]: ratio seems low — worth verifying spaced-email templates are being hit.")

    # -------------------------------------------------------------------------
    # Spot-check a few injected entries
    # -------------------------------------------------------------------------
    print("\n--- Sample injected entries (training) ---")
    train_email_entries = [e for e in train_prepared if 'B-EMAIL' in e['ner_tags']][:3]
    for i, s in enumerate(train_email_entries, 1):
        print(f"\n  [{i}] {s['sequence']}")
        for tok, tag in zip(s['tokens'], s['ner_tags']):
            if tag != 'O':
                print(f"       {tok!r:25s} {tag}")

    print("\n--- Sample injected entries (test) ---")
    test_email_entries = [e for e in test_prepared if 'B-EMAIL' in e['ner_tags']][:3]
    for i, s in enumerate(test_email_entries, 1):
        print(f"\n  [{i}] {s['sequence']}")
        for tok, tag in zip(s['tokens'], s['ner_tags']):
            if tag != 'O':
                print(f"       {tok!r:25s} {tag}")

    # Show at least one entry where the spaced format produced I-EMAIL tokens
    print("\n--- Example of a spaced email entry (B-EMAIL + I-EMAIL) ---")
    for e in train_prepared:
        if 'I-EMAIL' in e['ner_tags']:
            print(f"\n  {e['sequence']}")
            for tok, tag in zip(e['tokens'], e['ner_tags']):
                if tag != 'O':
                    print(f"       {tok!r:25s} {tag}")
            break

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    print("\nWriting prepared datasets to disk...")
    with open(DATA_DIR + 'train_prepared.json', 'w') as f:
        json.dump(train_prepared, f, indent=2)
    with open(DATA_DIR + 'test_prepared.json', 'w') as f:
        json.dump(test_prepared, f, indent=2)

   