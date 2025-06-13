import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim

from utils import FeatureDataset
from torch.utils.data import DataLoader

def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer

def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



def eval(model, data_loader, mode='backdoor', device='cpu'):
    model.eval()
    y_true = []
    y_predict = []
    for _, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
            
        _, batch_y_predict = torch.max(batch_y_predict, dim=1)

        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]

    return acc


def all_eval(model, test_loader, trfl, trrl, tefl, terl, device, head=None):
    print('*' * 100)
    print(' ' * 40 + 'Utility Evaluation')
    print('*' * 100)

    model.eval()
    test_acc = eval(model=model, data_loader=test_loader, device=device)
    forget_acc = eval(model=model, data_loader=tefl, device=device)
    remain_acc = eval(model=model, data_loader=terl, device=device)
    train_forget_acc = eval(model=model, data_loader=trfl, device=device)
    train_remain_acc = eval(model=model, data_loader=trrl, device=device)
    
    # print Utility Evaluation Result
    if head is None:
        print('-------Utility Evaluation Result-------')
        print('Train Forget Acc: {:.2%}'.format(train_forget_acc))
        print('Train Remain Acc: {:.2%}'.format(train_remain_acc))
        print('Forget Acc: {:.2%}'.format(forget_acc))
        print('Remain Acc: {:.2%}'.format(remain_acc))
        print('Test Acc: {:.2%}'.format(test_acc))
        print('---------------------------------------')
    else:
        print('-------Utility Evaluation Result-------')
        print(f"{head} Train Forget Acc: {train_forget_acc:.2%}")
        print(f"{head} Train Remain Acc: {train_remain_acc:.2%}")
        print(f"{head} Forget Acc: {forget_acc:.2%}")
        print(f"{head} Remain Acc: {remain_acc:.2%}")
        print(f"{head} Test Acc: {test_acc:.2%}")
        print('---------------------------------------')


def evaluate_KR(model, train_loader, test_loader, ttfl, ttrl, tefl, terl, num_classes, ckpt_dir=None, device="cuda", args=None):
    print('*' * 100)
    print(' ' * 30 + 'Knowledge Retention')
    print('*' * 100)
    model.eval()
    if args.model_name == 'AllCNN':
        model.head = nn.Linear(model.embed_dim, num_classes).to(device)
    else:
        model.head = nn.Linear(model.head.in_features, num_classes).to(device)
    nn.init.xavier_normal_(model.head.weight)
    nn.init.zeros_(model.head.bias)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = optimizer_picker('sgd', model.head.parameters(), lr=args.kr_lp)
    criterion = nn.CrossEntropyLoss()

    # for fast evaluation, we can use the feature loader directly
    # # get feature
    # features = []
    # labels = []

    # model.eval()
    # with torch.no_grad():
    #     for x, y in tqdm(train_loader):
    #         x, y = x.to(device), y.to(device)
            
    #         output = model(x, all=True)
    #         features.append(output['pre_logits'])
    #         labels.append(y)

    # features = torch.cat(features, dim=0)
    # labels = torch.cat(labels, dim=0)

    # feature_dataset = FeatureDataset(features, labels)
    # feature_loader = DataLoader(feature_dataset, batch_size=args.kr_batch_size, shuffle=True)

    # for epo in tqdm(range(args.kr_epoch)):
    #     for x, y in feature_loader:
    #         x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    #         logits = model.head(x)
    #         loss = criterion(logits, y)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    for epo in tqdm(range(args.kr_epoch)):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save model
    if ckpt_dir is not None:
        torch.save(model, '{}.pth'.format(ckpt_dir + "lp_model"))

    model.eval()
    with torch.no_grad():
        all_eval(model, test_loader, ttfl, ttrl, tefl, terl, device, head="KR")
