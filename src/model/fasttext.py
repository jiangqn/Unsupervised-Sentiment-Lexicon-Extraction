import torch
from torch import nn
import numpy as np
from src.utils.constants import PAD_INDEX

class FastText(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.output_projection = nn.Linear(embed_size, 2)

    def load_pretrained_embeddings(self, path, fix):
        self.embedding.weight.data.copy_(torch.from_numpy(np.load(path)))
        if fix:
            self.embedding.weight.requires_grad = False

    def forward(self, sentence):
        mask = (sentence != PAD_INDEX)
        sentence = self.embedding(sentence)
        sentence = sentence.masked_fill(mask.unsqueeze(-1) == 0, 0)
        sentence_lens = mask.float().sum(dim=1, keepdim=True)
        sentence = sentence.sum(dim=1, keepdim=False) / sentence_lens
        logit = self.output_projection(sentence)
        return logit