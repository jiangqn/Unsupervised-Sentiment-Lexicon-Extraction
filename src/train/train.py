import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import os
import pickle
from src.model.fasttext import FastText
from src.data_process.dataset import ClassifierDataset

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    base_path = './data/'
    processed_base_path = os.path.join(base_path, 'processed')
    processed_data_path = os.path.join(processed_base_path, 'data.npz')
    # word2index_path = os.path.join(processed_base_path, 'word2index.pkl')
    index2word_path = os.path.join(processed_base_path, 'index2word.pkl')
    glove_path = os.path.join(processed_base_path, 'glove.npy')
    save_path = os.path.join(processed_base_path, 'model.pkl')
    with open(index2word_path, 'rb') as handle:
        index2word = pickle.load(handle)
    model = FastText(vocab_size=len(index2word), embed_size=300)
    model.load_pretrained_embeddings(glove_path, fix=False)
    model = model.cuda()
    dataset = ClassifierDataset(processed_data_path)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    max_accuracy = 0
    for epoch in range(args.epoches):
        total_samples = 0
        total_loss = 0
        correct_samples = 0
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            sentence, label = data
            sentence, label = sentence.cuda(), label.cuda()
            logit = model(sentence)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            batch_size = label.size(0)
            total_samples += batch_size
            total_loss += batch_size * loss.item()
            pred = logit.argmax(dim=-1)
            correct_samples += (pred == label).long().sum().item()
            if i % 100 == 0:
                train_loss = total_loss / total_samples
                train_accuracy = correct_samples / total_samples
                print('[epoch %d] [step %d]\ttrain_loss: %.4f\ttrain_accuracy: %.4f' % (epoch, i, train_loss, train_accuracy))
                total_samples = 0
                total_loss = 0
                correct_samples = 0
                if train_accuracy > max_accuracy:
                    max_accuracy = train_accuracy
                    torch.save(model, save_path)