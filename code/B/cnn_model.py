"""CNN from EDA section 9.1, implemented in PyTorch."""

import hashlib

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, embedding, config):
        super().__init__()
        # The author feeds fixed GloVe vectors directly to Conv1D. A frozen
        # lookup table gives the same input representation without expanding
        # every sentence into a large float array in advance.
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding).clone(),
            freeze=config["freeze_embedding"],
            padding_idx=0,
        )
        self.conv = nn.Conv1d(
            config["embedding_dim"], config["conv_filters"], config["kernel_size"]
        )
        self.hidden = nn.Linear(config["conv_filters"], config["hidden_dim"])
        self.output = nn.Linear(config["hidden_dim"], config["num_classes"])
        # The paper states normal initialization for the network. The released
        # Keras code explicitly applies it only to its output layer; we follow
        # the paper for all trainable Conv/Linear weights and record this choice.
        for layer in (self.conv, self.hidden, self.output):
            nn.init.normal_(layer.weight, mean=0.0, std=config["init_std"])
            nn.init.zeros_(layer.bias)

    def forward(self, token_ids):
        features = self.embedding(token_ids).transpose(1, 2)
        features = torch.relu(self.conv(features)).amax(dim=2)
        features = torch.relu(self.hidden(features))
        # CrossEntropyLoss takes logits; apply softmax only when exporting
        # probabilities. This is the paper's softmax + categorical CE objective.
        return self.output(features)


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
