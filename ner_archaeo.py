import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

DATA_URL = "https://raw.githubusercontent.com/alexbrandsen/Archaeo-NER-data-English/main/5-folds-test-train-val-split/fold1/{}.txt"
SPLITS = ["train", "val", "test"]
ENTITY_TYPES = ["ART", "CON", "LOC", "MAT", "PER", "SPE"]
LABEL_NAMES = ["O"] + [f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}
FEATURES = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=LABEL_NAMES)),
})


# Data Download
def download_fold1(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = data_dir / f"{split}.txt"
        if not path.exists():
            urllib.request.urlretrieve(DATA_URL.format(split), path)


# IOB Conversion
def read_iob(path):
    sentences, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            word, _pos, label = line.split()
            tokens.append(word)
            tags.append(LABEL2ID[label])
        elif tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
            tokens, tags = [], []
    return Dataset.from_list(sentences, features=FEATURES)


def load_fold1(data_dir):
    return DatasetDict({split: read_iob(data_dir / f"{split}.txt") for split in SPLITS})


def tag_names(ids):
    return [LABEL_NAMES[i] for i in ids]


# Data Statistics
def print_statistics(dataset):
    train_forms = defaultdict(Counter)
    for split in SPLITS:
        tags = [tag_names(s["ner_tags"]) for s in dataset[split]]
        tokens = [s["tokens"] for s in dataset[split]]
        n_tokens = sum(len(t) for t in tags)
        n_outside = sum(t.count("O") for t in tags)
        entities = [(etype, " ".join(words[start:end + 1]).lower())
                    for words, sent_tags in zip(tokens, tags)
                    for etype, start, end in get_entities(sent_tags)]
        counts = Counter(etype for etype, _ in entities)
        no_entity = sum(1 for t in tags if set(t) == {"O"})
        print(f"\n{split}: {len(tags)} sentences, {n_tokens} tokens, {100 * n_outside / n_tokens:.1f}% O, "
              f"{no_entity} sentences without entities, {len(entities)} entities")
        for etype in ENTITY_TYPES:
            print(f"  {etype}: {counts[etype]:5d} ({100 * counts[etype] / len(entities):4.1f}%)")
        if split == "train":
            for etype, form in entities:
                train_forms[form][etype] += 1
        if split == "test":
            unseen = sum(1 for etype, form in entities if etype not in train_forms.get(form, {}))
            print(f"  test entities not seen with the same type in train: {unseen}/{len(entities)}")
    ambiguous = {f: dict(c) for f, c in train_forms.items() if len(c) > 1}
    print(f"\ntrain surface forms with more than one entity type: {len(ambiguous)}/{len(train_forms)}")
    for form, counts in sorted(ambiguous.items(), key=lambda x: -sum(x[1].values()))[:5]:
        print(f"  {form}: {counts}")


# Label Alignment
def tokenize_and_align_labels(batch, tokenizer):
    encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    encoded["labels"] = []
    for i, word_tags in enumerate(batch["ner_tags"]):
        labels, previous = [], None
        for word_id in encoded.word_ids(i):
            # only the first subword of each word carries the label, so evaluation runs on whole words
            labels.append(-100 if word_id is None or word_id == previous else word_tags[word_id])
            previous = word_id
        encoded["labels"].append(labels)
    return encoded


# Evaluation Helpers
def to_tag_lists(predictions, label_ids):
    gold, pred = [], []
    for pred_row, label_row in zip(predictions, label_ids):
        keep = label_row != -100
        gold.append(tag_names(label_row[keep]))
        pred.append(tag_names(pred_row[keep]))
    return gold, pred


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    gold, pred = to_tag_lists(logits.argmax(-1), label_ids)
    return {
        "precision": precision_score(gold, pred),
        "recall": recall_score(gold, pred),
        "f1": f1_score(gold, pred),
        "accuracy": accuracy_score(gold, pred),
    }


# Model Training
def train_and_predict(dataset, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id=LABEL2ID,
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_model"))

    output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    print("\nTest set:", output.metrics)
    _, pred_tags = to_tag_lists(output.predictions.argmax(-1), output.label_ids)
    return [
        {"tokens": s["tokens"], "gold": tag_names(s["ner_tags"]), "pred": pred + ["O"] * (len(s["tokens"]) - len(pred))}
        for s, pred in zip(dataset["test"], pred_tags)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/fold1"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", default="bert-base-cased")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    download_fold1(args.data_dir)
    dataset = load_fold1(args.data_dir)
    print(dataset)
    print_statistics(dataset)
    if args.stats_only:
        return

    predictions = train_and_predict(dataset, args.model, args.output_dir)
    out_path = args.output_dir / "test_predictions.json"
    out_path.write_text(json.dumps(predictions, indent=1), encoding="utf-8")
    print(f"Saved word-level gold and predicted tags for {len(predictions)} test sentences to {out_path}")


if __name__ == "__main__":
    main()
