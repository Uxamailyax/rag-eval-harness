import json
from collections import Counter

strata = {
    json.loads(l)["case_id"]: json.loads(l)["stratum"]
    for l in open("evals/labelset_v1.jsonl", encoding="utf-8")
}
labels = [json.loads(l) for l in open("evals/human_labels_v1.jsonl", encoding="utf-8")]

tally = Counter()
for x in labels:
    s = strata[x["case_id"]]
    print(f"{x['case_id']}  {s:<10} label={x['faithful']}")
    tally[(s, x["faithful"])] += 1

print()
for (s, lab), n in sorted(tally.items()):
    print(f"{s:<10} labelled {lab}: {n}")