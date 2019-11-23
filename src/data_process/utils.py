import os
import numpy as np
from src.utils.constants import PAD_INDEX, INIT

def load_glove(path, vocab_size, word2index):
    if not os.path.isfile(path):
        raise IOError('Not a file', path)
    glove = np.random.uniform(-INIT, INIT, [vocab_size, 300])
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                glove[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    glove[PAD_INDEX, :] = 0
    return glove