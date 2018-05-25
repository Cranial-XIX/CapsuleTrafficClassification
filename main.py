import argparse
import config
import numpy as np
import pickle
import torch

from torch.optim import Adam
from model import ConvNet, CapsuleNet

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn', help='cnn | capsule')
parser.add_argument('--seed', default=0, help='random seed')
args = parser.parse_args()

print("Using model", args.model)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, X_tr, y_tr):
    model.train()
    i = np.random.permutation(len(y_tr))
    num_batch = len(y_tr) // config.BATCH_SIZE
    X_split, y_split = np.split(X_tr[i], num_batch), np.split(y_tr[i], num_batch)
    tot_loss = 0.0
    for bch, (X, y) in enumerate(zip(X_split, y_split)):
        optimizer.zero_grad()
        X = torch.from_numpy(X).float().permute(0, 3, 1, 2).to(device)
        y = torch.from_numpy(y).to(device)
        loss = model(X, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        print(" batch {} [{:.1f}%] loss: {:.4f}".format(
            bch+1, 100 * (bch+1)/num_batch, loss.item()/len(y)))
    tot_loss /= len(y_tr)
    print(" Total loss %.2f" % (tot_loss))
    return tot_loss


def test(model, X_te, y_te, mode):
    model.eval()
    num_batch = len(y_te) // config.BATCH_SIZE
    X_split, y_split = np.split(X_te, num_batch), np.split(y_te, num_batch)
    accuracy = 0.0
    for X, y in zip(X_split, y_split): 
        X = torch.from_numpy(X).float().permute(0, 3, 1, 2).to(device)
        prediction = model(X)
        accuracy += np.sum(y == prediction.cpu().numpy())
    accuracy /= len(y_te)
    print("%s Accuracy  %.2f" % (mode, accuracy))
    return accuracy


def load_data():
    X_tr, y_tr = pickle.load(open('data/train.p', 'rb'))
    X_te, y_te = pickle.load(open('data/test.p', 'rb'))
    return X_tr, y_tr, X_te, y_te


def main():
    X_tr, y_tr, X_te, y_te = load_data()
    X_tr, y_tr = X_tr[:1024], y_tr[:1024]
    X_te, y_te = X_te[:128], y_te[:128]
    model = ConvNet() if args.model == 'cnn' else CapsuleNet()
    model.to(device)
    optimizer = Adam(model.parameters())
    train_loss = []
    train_accuracy = []
    for epoch in range(10):
        print(("Epoch %d " + "-"*70) % (epoch+1))
        train_loss.append(train(model, optimizer, X_tr, y_tr))
        train_accuracy.append(test(model, X_tr, y_tr, "Train"))
    pickle.dump((train_loss, train_accuracy), \
        open('result/' + args.model + '_train.p', 'wb'))

if __name__ == "__main__":
    main()