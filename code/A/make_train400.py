import numpy as np
from sklearn.model_selection import train_test_split

with open("data/sst2_train_500.txt", encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f if l.strip()]
labels = [int(l.split("\t")[0]) for l in lines]

tr, _ = train_test_split(np.arange(500), test_size=0.2, stratify=labels, random_state=42)
tr = sorted(tr.tolist())

with open("data/train_400.txt", "w", encoding="utf-8") as f:
    for i in tr:
        f.write(lines[i] + "\n")

print("train_400 lines:", len(tr))
print("train_400 label distribution:", {k: sum(1 for i in tr if labels[i] == k) for k in (0, 1)})