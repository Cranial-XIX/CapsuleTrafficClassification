import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from torch.optim import Adam

BATCH_SIZE = 8
NUM_CLASSES = 43

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        conv = self.conv(x)
        scores = self.fc2(self.relu(self.fc1(conv.view(64, -1))))
        if y is None:
            _, prediction = scores.max(dim=1)
            return prediction
        return (-self.lsm(scores).gather(1, y.unsqueeze(1))).sum()


class CapsuleLayer(nn.Module):
    def __init__(self, n_caps, n_nodes, in_C, out_C,
        kernel=None, stride=None, n_iter=3):
        super(CapsuleLayer, self).__init__()

        self.n_iter = n_iter
        self.n_nodes = n_nodes
        self.n_caps = n_caps

        self.softmax = nn.Softmax(dim=2)

        if n_nodes != -1: # caps -> caps layer
            self.route_weights = nn.Parameter(
                torch.randn(1, n_nodes, n_caps, in_C, out_C))
        else:   # conv -> caps layer
            self.capsules = nn.ModuleList([nn.Conv2d(
                in_C, out_C, kernel, stride=stride
            ) for _ in range(n_caps)])

    def squash(self, v):
        squared_norm = (v ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * v / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.n_nodes != -1:
            priors = x[:, :, None, None, :] @ self.route_weights
            logits = torch.zeros(*priors.size())
            # dynamic routing
            for i in range(self.n_iter):
                probs = self.softmax(logits)
                outputs = self.squash((probs * priors).sum(dim=1, keepdim=True))
                if i != self.n_iter - 1:
                    delta = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta
        else:
            outputs = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
            outputs = self.squash(torch.cat(outputs, dim=-1))
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.conv1 = nn.Conv2d(3, 256, 9)
        self.primary_capsules = CapsuleLayer(
            n_caps=8, n_nodes=-1, in_C=256, out_C=32, kernel=8, stride=2)
        self.traffic_sign_capsules = CapsuleLayer(
            n_caps=NUM_CLASSES, n_nodes=32 * 9 * 9, in_C=8, out_C=16)

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.traffic_sign_capsules(x).squeeze()

        scores = (x ** 2).sum(dim=-1) ** 0.5
        if y is None:
            # In all batches, get the most active capsule.
            _, prediction = scores.max(dim=1)
            return prediction

        left = self.relu(0.9 - scores) ** 2
        right = self.relu(scores - 0.1) ** 2
        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=y)
        margin_loss = labels * left + 0.5 * (1. - labels) * right

        return margin_loss.sum() / y.size(0)


def train(model, optimizer, X_tr, y_tr):
    model.train()
    i = np.random.permutation(len(y_tr))
    num_batch = len(y_tr) // BATCH_SIZE
    X_split, y_split = np.split(X_tr[i], num_batch), np.split(y_tr[i], num_batch)
    tot_loss = 0.0
    for bch, (X, y) in enumerate(zip(X_split, y_split)):
        optimizer.zero_grad()
        loss = model(
            torch.from_numpy(X).float().permute(0, 3, 1, 2),
            torch.from_numpy(y))
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        print(" batch {} [{:.1f}%] loss: {:.2f}".format(
            bch+1, 100 * (bch+1)/num_batch, loss.item()/len(y)))
    print(" Total loss %.2f" % (tot_loss/len(y_tr)))


def test(model, X_te, y_te, mode):
    model.eval()
    num_batch = len(y_te) // BATCH_SIZE
    X_split, y_split = np.split(X_te, num_batch), np.split(y_te, num_batch)
    accuracy = 0.0
    for X, y in zip(X_split, y_split):
        prediction = model(torch.from_numpy(X).float().permute(0, 3, 1, 2))
        print(y)
        print(prediction.numpy())
        accuracy += np.sum(y == prediction.numpy())
    print("%s Accuracy  %.2f" % (mode, accuracy/len(y_te)))


def load_data():
    X_tr, y_tr = pickle.load(open('data/train.p', 'rb'))
    X_te, y_te = pickle.load(open('data/test.p', 'rb'))
    return X_tr, y_tr, X_te, y_te


def main():
    X_tr, y_tr, X_te, y_te = load_data()
    X_tr, y_tr = X_tr[:8], y_tr[:8]
    X_te, y_te = X_te[:128], y_te[:128]
    convnet = CapsuleNet()#ConvNet()
    optimizer = Adam(convnet.parameters(), lr=1e-4)
    for epoch in range(30):
        print(("Epoch %d " + "-"*70) % (epoch+1))
        train(convnet, optimizer, X_tr, y_tr)
        #test(convnet, X_te, y_te, "Test")
        test(convnet, X_tr, y_tr, "Train")

if __name__ == "__main__":
    main()

