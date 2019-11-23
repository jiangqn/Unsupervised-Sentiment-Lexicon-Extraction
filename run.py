import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='preprocess', choices=['preprocess', 'train', 'extract', 'replace'])
parser.add_argument('--epoches', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3])

args = parser.parse_args()

if args.task == 'preprocess':
    from src.data_process.preprocess import preprocess
    preprocess()
elif args.task == 'train':
    from src.train.train import train
    train(args)
elif args.task == 'extract':
    from src.data_process.extract import extract
    extract(args)
elif args.task == 'replace':
    from src.data_process.replace import replace
    replace()