import os
import numpy as np
import pickle
from src.utils.constants import SPLIT, UNK
from src.data_process.vocab import Vocab
from src.data_process.utils import load_glove

def preprocess():
    base_path = './data/'
    raw_path = os.path.join(base_path, 'raw/data.txt')
    processed_base_path = os.path.join(base_path, 'processed')
    processed_data_path = os.path.join(processed_base_path, 'data.npz')
    word2index_path = os.path.join(processed_base_path, 'word2index.pkl')
    index2word_path = os.path.join(processed_base_path, 'index2word.pkl')
    glove_source_path = '../datasets/embeddings/glove.840B.300d.txt'
    glove_path = os.path.join(processed_base_path, 'glove.npy')

    if not os.path.exists(processed_base_path):
        os.makedirs(processed_base_path)

    raw_file = open(raw_path, 'r', encoding='utf-8').readlines()

    d = {
        'negative': 0,
        'positive': 1
    }

    vocab = Vocab()

    sentences = []
    labels = []

    max_len = 0

    for line in raw_file:
        label, sentence = line.strip().split(SPLIT)
        labels.append(d[label])
        sentence = sentence.split()
        vocab.add_list(sentence)
        sentences.append(sentence)
        max_len = max(max_len, len(sentence))

    word2index, index2word = vocab.get_vocab()

    num = len(sentences)

    f = lambda x: word2index[x] if x in word2index else word2index[UNK]

    for i in range(num):
        sentences[i] = [f(word) for word in sentences[i]]
        sentences[i].extend([0] * (max_len - len(sentences[i])))

    sentences = np.asarray(sentences, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    np.savez(processed_data_path, sentence=sentences, label=labels)

    with open(word2index_path, 'wb') as handle:
        pickle.dump(word2index, handle)
    with open(index2word_path, 'wb') as handle:
        pickle.dump(index2word, handle)

    glove = load_glove(glove_source_path, len(index2word), word2index)
    np.save(glove_path, glove)