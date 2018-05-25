import config
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc2 = nn.Linear(128, config.NUM_CLASSES)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        conv = self.conv(x)
        scores = self.fc2(
            self.relu(self.fc1(conv.view(config.BATCH_SIZE, -1))))
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
            self.route_weights = nn.Parameter(0.1 *
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
            priors = (x[:, :, None, None, :] @ self.route_weights).squeeze(4)
            logits = torch.zeros(*priors.size()).to(device)
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
            n_caps=config.NUM_CLASSES, n_nodes=32 * 9 * 9, in_C=8, out_C=16)

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
        labels = torch.eye(config.NUM_CLASSES).to(
            device).index_select(dim=0, index=y)
        margin_loss = labels * left + 0.5 * (1. - labels) * right

        return margin_loss.sum() / y.size(0)