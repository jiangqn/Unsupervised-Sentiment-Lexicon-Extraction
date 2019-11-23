import os
from src.utils.constants import SPLIT

def replace():
    base_path = './data/'
    raw_path = os.path.join(base_path, 'raw/data.txt')
    processed_base_path = os.path.join(base_path, 'processed')
    save_path = os.path.join(processed_base_path, 'unparallel.txt')
    sentiment_lexicon_path = os.path.join(processed_base_path, 'sentiment_lexicon.txt')
    sentiment_lexicon_file = open(sentiment_lexicon_path, 'r', encoding='utf-8')
    sentiment_lexicon = set()
    for line in sentiment_lexicon_file.readlines():
        sentiment_lexicon.add(line.strip())
    raw_file = open(raw_path, 'r', encoding='utf-8')
    save_file = open(save_path, 'w', encoding='utf-8')
    for i, line in enumerate(raw_file.readlines()):
        label, sentence = line.strip().split(SPLIT)
        sentence = sentence.split()
        text = ''
        for word in sentence:
            if word in sentiment_lexicon:
                text += '<s> '
            else:
                text += word + ' '
        text = text.strip()
        save_file.write(text + '\n')
        if i % 10000 == 0:
            print(i)