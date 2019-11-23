import os
import torch
import pickle

def extract(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    base_path = './data/'
    processed_base_path = os.path.join(base_path, 'processed')
    model_path = os.path.join(processed_base_path, 'model.pkl')
    model = torch.load(model_path)
    index2word_path = os.path.join(processed_base_path, 'index2word.pkl')
    with open(index2word_path, 'rb') as handle:
        index2word = pickle.load(handle)
    save_path = os.path.join(processed_base_path, 'sentiment_lexicon.txt')
    vocab_size = len(index2word)
    words = torch.arange(0, vocab_size).long().unsqueeze(-1).cuda()
    model.eval()
    with torch.no_grad():
        logits = model(words)
        logits[0] = 0
        diff = logits[:, 1] - logits[:, 0]
    diff, index = diff.sort()
    diff = diff.tolist()
    index = index.tolist()
    sentiment_lexicon = []
    for i in range(500):
        # print(index2word[index[i]], diff[i])
        sentiment_lexicon.append(index2word[index[i]])
    # print('')
    for i in range(500):
        # print(index2word[index[-i-1]], diff[-i-1])
        sentiment_lexicon.append(index2word[index[-i-1]])
    save_file = open(save_path, 'w', encoding='utf-8')
    for word in sentiment_lexicon:
        save_file.write(word + '\n')