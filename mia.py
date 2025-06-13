import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import numpy as np  
from sklearn import linear_model, model_selection

from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit


def compute_losses(net, loader, device):
    """Auxiliary function to compute per-sample losses"""
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    net.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = net(inputs)
            # logits = logits['pre_logits']
            # logits = logits['logits']
            pre = torch.argmax(logits, dim=1)
            losses = criterion(logits, targets).detach().cpu().numpy()
            for l in losses:
                all_losses.append(l)

    return np.array(all_losses)

class ComplexDNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def train_model(model, optimizer, criterion, loader):
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(-1, 1).float())
        loss.backward()
        optimizer.step()

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = (output > 0.5).long()
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(loader.dataset)
    return accuracy

def simple_mia_pytorch(sample_loss, members, n_splits=5, random_state=0, epochs=30, batch_size=32):
    sample_loss_tensor = torch.Tensor(sample_loss.reshape(-1, 1))
    members_tensor = torch.Tensor(members)

    sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    scores = []

    for train_idx, test_idx in sss.split(sample_loss_tensor, members_tensor):
        X_train, X_test = sample_loss_tensor[train_idx], sample_loss_tensor[test_idx]
        y_train, y_test = members_tensor[train_idx], members_tensor[test_idx]

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = ComplexDNN(input_dim=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            train_model(model, optimizer, criterion, train_loader)
        
        accuracy = evaluate_model(model, test_loader)
        scores.append(accuracy)

    return np.array(scores)

def simple_mia(sample_loss, members, n_splits=5, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def evaluate_mia(model, trfl, tefl, device, args):
    print('*' * 100)
    print(' ' * 20 + 'Membership Inference Attack')
    print('*' * 100)

    forget_losses = compute_losses(model, trfl, device)
    test_losses = compute_losses(model, tefl, device)

    if len(forget_losses) > len(test_losses):
        np.random.shuffle(forget_losses)
        forget_losses = forget_losses[: len(test_losses)]
    else:
        np.random.shuffle(test_losses)
        test_losses = test_losses[: len(forget_losses)]

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

    if args.use_pytorch_mia:
        mia_scores = simple_mia_pytorch(samples_mia, labels_mia)
    else:
        mia_scores = simple_mia(samples_mia, labels_mia)
    # print(
    #     f"The MIA has an accuracy of {mia_scores.mean():.4f} on forgotten vs unseen images"
    # )
    mia_scores = mia_scores * 100
    print(f"MIA: {mia_scores.mean():.2f}")
