import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2, Entities
from sklearn.metrics import precision_recall_fscore_support

ENTITY_TYPES = ["ART", "CON", "LOC", "MAT", "PER", "SPE"]


# Data Load
def load_predictions(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r["tokens"] for r in rows], [r["gold"] for r in rows], [r["pred"] for r in rows]


def read_iob(path):
    sentences, words, tags = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            word, _pos, tag = line.split()
            words.append(word)
            tags.append(tag)
        elif words:
            sentences.append((words, tags))
            words, tags = [], []
    return sentences


def strict_entities(tags):
    return {(e.tag, e.start, e.end - 1) for e in Entities([tags], IOB2).entities[0]}


def phrase(words, start, end):
    return " ".join(words[start:end + 1]).lower()


# Token Labels
def token_scores(gold, pred, prefix):
    labels = [f"{prefix}-{etype}" for etype in ENTITY_TYPES]
    flat_gold = [t for sent in gold for t in sent]
    flat_pred = [t for sent in pred for t in sent]
    p, r, f, n = precision_recall_fscore_support(flat_gold, flat_pred, labels=labels, zero_division=0)
    scores = {etype: (p[i], r[i], f[i], n[i]) for i, etype in enumerate(ENTITY_TYPES)}
    p, r, f, _ = precision_recall_fscore_support(flat_gold, flat_pred, labels=labels, average="micro", zero_division=0)
    scores["micro"] = (p, r, f, sum(n))
    present = [s for etype, s in scores.items() if etype in ENTITY_TYPES and s[3] > 0]
    scores["macro"] = tuple(sum(s[k] for s in present) / len(present) for k in range(3)) + (sum(n),)
    return scores


# Full Entities
def entity_scores(gold, pred):
    report = classification_report(gold, pred, output_dict=True, mode="strict", scheme=IOB2, zero_division=0)
    keys = {etype: etype for etype in ENTITY_TYPES} | {"micro": "micro avg", "macro": "macro avg"}
    return {name: tuple(report[k][m] for m in ("precision", "recall", "f1-score", "support"))
            for name, k in keys.items()}


# Error Analysis
def overlaps(a, b):
    return a[1] <= b[2] and b[1] <= a[2]


def error_analysis(tokens, gold, pred, train_forms):
    pred_errors, gold_outcomes, seen, by_length = Counter(), Counter(), Counter(), Counter()
    spans, shapes = Counter(), Counter()
    b_correct = b_correct_wrong_span = 0
    for words, g_tags, p_tags in zip(tokens, gold, pred):
        g_ents, p_ents = strict_entities(g_tags), strict_entities(p_tags)
        for p in p_ents - g_ents:
            hits = [g for g in g_ents if overlaps(p, g)]
            if not hits:
                pred_errors["spurious"] += 1
            elif any(g[1:] == p[1:] for g in hits):
                pred_errors["wrong type, same span"] += 1
            elif any(g[0] == p[0] for g in hits):
                pred_errors["right type, wrong boundary"] += 1
            else:
                pred_errors["wrong type and boundary"] += 1
        for g in g_ents:
            exact = g in p_ents
            gold_outcomes["exact" if exact else "partial" if any(overlaps(g, p) for p in p_ents) else "missed"] += 1
            seen[(g[0], (g[0], phrase(words, g[1], g[2])) in train_forms, exact)] += 1
            by_length[(min(g[2] - g[1] + 1, 3), exact)] += 1
            if p_tags[g[1]] == g_tags[g[1]]:
                b_correct += 1
                b_correct_wrong_span += not exact
            if not exact and g[2] > g[1]:
                inside = p_tags[g[1]:g[2] + 1]
                spans[(phrase(words, g[1], g[2]), " ".join(g_tags[g[1]:g[2] + 1]), " ".join(inside))] += 1
                shapes[(g[0], "missed entirely" if all(t == "O" for t in inside)
                        else "first word only" if inside[0] == f"B-{g[0]}" and all(t == "O" for t in inside[1:])
                        else "last word only" if inside[-1] == f"B-{g[0]}" and all(t == "O" for t in inside[:-1])
                        else "other")] += 1
    return pred_errors, gold_outcomes, seen, by_length, (b_correct, b_correct_wrong_span), spans, shapes


# Error Details
def word_labels(sentences):
    labels = defaultdict(Counter)
    for words, tags in sentences:
        for word, tag in zip(words, tags):
            labels[word.lower()][tag] += 1
    return labels


def phrase_patterns(sentences, text):
    singular = lambda ws: " ".join(ws).rstrip("s")
    target, n, patterns = singular(text.split()), len(text.split()), Counter()
    for words, tags in sentences:
        lowered = [w.lower() for w in words]
        for i in range(len(lowered) - n + 1):
            if singular(lowered[i:i + n]) == target:
                patterns[" ".join(tags[i:i + n])] += 1
    return patterns


def print_details(tokens, gold, pred, train):
    flat = [(w.lower(), g, p) for ws, gs, ps in zip(tokens, gold, pred) for w, g, p in zip(ws, gs, ps)]
    errors = Counter((g, p) for _, g, p in flat if g != p)
    print("\nMost frequent wrong predictions per gold label:")
    for label in ["O"] + [f"{x}-{etype}" for etype in ENTITY_TYPES for x in "BI"]:
        wrong = Counter({p: n for (g, p), n in errors.items() if g == label})
        if wrong:
            print(f"  {label:6} ({sum(1 for _, g, _ in flat if g == label)} tokens): {wrong.most_common(3)}")

    test_labels, train_labels = word_labels(zip(tokens, gold)), word_labels(train)
    print("\nWords behind B-label errors (missed: gold B-X predicted otherwise; false: gold O predicted B-X):")
    for etype in ENTITY_TYPES:
        label = f"B-{etype}"
        missed = Counter(w for w, g, p in flat if g == label != p)
        false = Counter(w for w, g, p in flat if g == "O" and p == label)
        print(f"  {etype}: missed {sum(missed.values())} {missed.most_common(3)}; false {sum(false.values())} {false.most_common(3)}")
        for word, _ in false.most_common(3):
            print(f"      '{word}' labelled O: test {test_labels[word]['O']}/{sum(test_labels[word].values())}, "
                  f"train {train_labels[word]['O']}/{sum(train_labels[word].values())}")


# Train Annotation
def print_train_annotation(train, spans, shapes):
    total, multi = Counter(), Counter()
    for _, tags in train:
        for etype, start, end in strict_entities(tags):
            total[etype] += 1
            multi[etype] += end > start
    print("\nTrain entities longer than one word: " + ", ".join(f"{t} {100 * multi[t] / total[t]:.0f}%" for t in ENTITY_TYPES))
    print("\nWrongly predicted multi-word test entities by what the model predicted inside them:")
    for etype in ENTITY_TYPES:
        counts = {shape: n for (t, shape), n in shapes.items() if t == etype}
        if counts:
            print(f"  {etype}: {sum(counts.values())} {counts}")
    print("\nMost frequent wrongly predicted multi-word test entities, with their annotation in train:")
    for (text, gold_tags, pred_tags), n in spans.most_common(10):
        print(f"  '{text}' {n}x: gold {gold_tags}, pred {pred_tags}; train {dict(phrase_patterns(train, text))}")


# Output
def print_table(title, scores):
    print(f"\n{title}\n{'':8}{'P':>7}{'R':>7}{'F1':>7}{'n':>7}")
    for name, (p, r, f, n) in scores.items():
        print(f"{name:8}{p:7.3f}{r:7.3f}{f:7.3f}{n:7d}")


def latex_table(b_scores, i_scores, e_scores):
    fmt = lambda s, with_n=True: " & ".join(f"{v:.3f}" for v in s[:3]) + (f" & {s[3]}" if with_n else "")
    rows = []
    for name in ENTITY_TYPES + ["micro", "macro"]:
        i_cells = fmt(i_scores[name]) if i_scores[name][3] > 0 else "-- & -- & -- & 0"
        label = name if name in ENTITY_TYPES else f"{name} avg."
        rows.append(f"{label} & {fmt(b_scores[name])} & {i_cells} & {fmt(e_scores[name], False)} \\\\")
        if name == ENTITY_TYPES[-1]:
            rows.append("\\midrule")
    header = (
        "\\begin{tabular}{l rrrr rrrr rrr}\n\\toprule\n"
        " & \\multicolumn{4}{c}{B- label} & \\multicolumn{4}{c}{I- label} & \\multicolumn{3}{c}{Full entity} \\\\\n"
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-12}\n"
        "Type & P & R & F1 & $n$ & P & R & F1 & $n$ & P & R & F1 \\\\\n\\midrule\n"
    )
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("output/test_predictions.json"))
    parser.add_argument("--train", type=Path, default=Path("data/fold1/train.txt"))
    parser.add_argument("--latex", type=Path, default=Path("output/results_table.tex"))
    args = parser.parse_args()

    tokens, gold, pred = load_predictions(args.predictions)
    b_scores, i_scores = token_scores(gold, pred, "B"), token_scores(gold, pred, "I")
    e_scores = entity_scores(gold, pred)
    print_table("B- labels (token level)", b_scores)
    print_table("I- labels (token level)", i_scores)
    print_table("Full entities (strict IOB2: all tokens of the entity correct)", e_scores)
    without_spe = [e_scores[t][2] for t in ENTITY_TYPES if t != "SPE"]
    print(f"\nEntity macro F1 without SPE: {sum(without_spe) / len(without_spe):.3f}")
    loose_i = sum(t.startswith("I-") and prev[2:] != t[2:] for ps in pred for prev, t in zip(["O"] + ps, ps))
    print(f"Micro F1 in seqeval default mode, as printed during training: {f1_score(gold, pred):.3f} "
          f"(that mode also counts the {loose_i} predicted I- tags that do not continue an entity as chunks)")
    args.latex.write_text(latex_table(b_scores, i_scores, e_scores), encoding="utf-8")
    print(f"\nLaTeX table written to {args.latex}")

    train = read_iob(args.train)
    train_forms = Counter((etype, phrase(words, s, e)) for words, tags in train for etype, s, e in strict_entities(tags))
    pred_errors, gold_outcomes, seen, by_length, b_counts, spans, shapes = error_analysis(tokens, gold, pred, train_forms)
    print(f"\nWrong predicted entities: {dict(pred_errors)}")
    print(f"Gold entities: {dict(gold_outcomes)}")
    print(f"Gold entities with a correct B-label: {b_counts[0]}, of which wrong span: {b_counts[1]}")
    print("Exact matches by entity length: " + ", ".join(
        f"{'3+' if n == 3 else n} words {by_length[(n, True)]}/{by_length[(n, True)] + by_length[(n, False)]}"
        for n in (1, 2, 3)))
    print(f"\n{'':6}{'seen in train':>22}{'unseen':>18}")
    for etype in ENTITY_TYPES:
        cells = []
        for is_seen in (True, False):
            found, total = seen[(etype, is_seen, True)], seen[(etype, is_seen, True)] + seen[(etype, is_seen, False)]
            cells.append(f"{found:>4}/{total:<4} R={found / total:.2f}" if total else f"{'0/0':>18}")
        print(f"{etype:6}{cells[0]:>22}{cells[1]:>18}")

    print_details(tokens, gold, pred, train)
    print_train_annotation(train, spans, shapes)


if __name__ == "__main__":
    main()
