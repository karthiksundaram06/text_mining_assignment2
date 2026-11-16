import json
from sklearn.metrics import precision_recall_fscore_support as prfs
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

data = json.load(open("test_predictions.json"))
gold = [x["gold"] for x in data]
pred = [x["pred"] for x in data]
flat_g = [l for s in gold for l in s]
flat_p = [l for s in pred for l in s]
types = ["ART", "CON", "LOC", "MAT", "PER", "SPE"]
def row(name, p, r, f, n=""):
    print(f"{name:12}{p:7.3f}{r:7.3f}{f:7.3f}{n:>7}")

# 1. Token level: B- and I- labels per type, plus micro/macro per prefix
print("TOKEN LEVEL")
for prefix in ("B", "I"):
    labs = [f"{prefix}-{t}" for t in types if f"{prefix}-{t}" in set(flat_g)]
    p, r, f, n = prfs(flat_g, flat_p, labels=labs, zero_division=0)
    for l, a, b, c, s in zip(labs, p, r, f, n):
        row(l, a, b, c, s)
    for avg in ("micro", "macro"):
        a, b, c, _ = prfs(flat_g, flat_p, labels=labs, average=avg, zero_division=0)
        row(f"{prefix} {avg}", a, b, c, int(n.sum()))

# 2. Full entities: strict IOB2 matching (all tokens must be right)
print("\nENTITY LEVEL (strict IOB2)")
rep = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
for t in types:
    v = rep[t]
    row(t, v["precision"], v["recall"], v["f1-score"], int(v["support"]))
for avg in ("micro avg", "macro avg"):
    v = rep[avg]
    row(avg, v["precision"], v["recall"], v["f1-score"], int(v["support"]))