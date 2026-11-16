import os
import numpy as np

PAD, UNK = "<PAD>", "<UNK>"
MAX_LEN = 50
GLOVE_PATH = os.environ.get(
    "GLOVE_PATH",
    r"C:\Users\11348\Desktop\glove\glove.6B.300d.txt",
)

def read_tsv(path):
    labels, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            labels.append(int(parts[0]))
            texts.append(parts[1])
    return texts, labels

def read_glue_dev(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            texts.append(parts[0])
            labels.append(int(parts[1]))
    return texts, labels

def build_vocab(texts):
    vocab = {PAD: 0, UNK: 1}
    for s in texts:
        for w in s.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def load_glove(path, vocab, dim=300):
    emb = np.random.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    emb[vocab[PAD]] = 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            if w in vocab:
                emb[vocab[w]] = np.asarray(parts[1:], dtype="float32")
    return emb

def encode(texts, vocab, max_len=MAX_LEN):
    ids = []
    for s in texts:
        seq = [vocab.get(w, vocab[UNK]) for w in s.split()][:max_len]
        seq += [vocab[PAD]] * (max_len - len(seq))
        ids.append(seq)
    return np.array(ids, dtype="int64")

def load_splits(data_dir="data", glove_path=GLOVE_PATH):
    base_texts, base_y = read_tsv(os.path.join(data_dir, "train_400.txt"))
    eda_texts, eda_y = read_tsv(os.path.join(data_dir, "train_eda_final.txt"))
    val_texts, val_y = read_tsv(os.path.join(data_dir, "val_100.txt"))
    test_texts, test_y = read_glue_dev(os.path.join(data_dir, "sst2_dev_872.tsv"))

    vocab = build_vocab(eda_texts)
    emb = load_glove(glove_path, vocab)

    return {
        "vocab": vocab,
        "embedding": emb,
        "baseline_train_x": encode(base_texts, vocab),
        "baseline_train_y": np.array(base_y, dtype="int64"),
        "train_x": encode(eda_texts, vocab),
        "train_y": np.array(eda_y, dtype="int64"),
        "val_x": encode(val_texts, vocab),
        "val_y": np.array(val_y, dtype="int64"),
        "test_x": encode(test_texts, vocab),
        "test_y": np.array(test_y, dtype="int64"),
        "baseline_train_texts": base_texts,
        "train_texts": eda_texts,
        "val_texts": val_texts,
        "test_texts": test_texts,
    }