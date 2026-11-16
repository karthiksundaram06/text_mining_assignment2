import json
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

data = json.load(open("test_predictions.json"))
gold = [x["gold"] for x in data]
pred = [x["pred"] for x in data]

# 1. Token-level scores per B-/I- label (O excluded)
flat_g = [l for s in gold for l in s]
flat_p = [l for s in pred for l in s]
labels = sorted({l for l in flat_g + flat_p if l != "O"}, key=lambda l: (l[2:], l[0]))
p, r, f, sup = precision_recall_fscore_support(flat_g, flat_p, labels=labels, zero_division=0)
print("\nTOKEN LEVEL (per B-/I- label)")
print(f"{'label':8}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
for l, a, b, c, s in zip(labels, p, r, f, sup):
    print(f"{l:8}{a:7.3f}{b:7.3f}{c:7.3f}{s:9d}")

# 2. Full-entity scores per entity type (strict span match)
print("\nENTITY LEVEL (per type, strict match)")
rep = classification_report(gold, pred, digits=3, output_dict=True, zero_division=0)
print(f"{'type':8}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
for t in sorted(k for k in rep if k not in ("micro avg", "macro avg", "weighted avg")):
    v = rep[t]
    print(f"{t:8}{v['precision']:7.3f}{v['recall']:7.3f}{v['f1-score']:7.3f}{v['support']:9d}")

# 3. Overall entity-level micro/macro F1
print("\nOVERALL")
print(f"Micro P/R/F1: {precision_score(gold,pred):.3f} / {recall_score(gold,pred):.3f} / {f1_score(gold,pred):.3f}")
print(f"Macro F1 (all types): {rep['macro avg']['f1-score']:.3f}")