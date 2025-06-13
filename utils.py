import os
import random
import torch

from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import shutil
import tarfile

import numpy as np


class CustomDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.targets = original_dataset.targets
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        data, label = self.original_dataset[index]
        return data, label


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class FeatureDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def get_dataset(data_name, path='./data', size_scale_ratio=None):

    if (data_name == 'mnist'):
        trainset = datasets.MNIST(path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        testset = datasets.MNIST(path, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

    # model: ResNet-50
    elif (data_name == 'cifar10'):
        transform = [transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if size_scale_ratio is not None:
            transform = [transforms.RandomResizedCrop(size_scale_ratio[0], scale=size_scale_ratio[1], ratio=size_scale_ratio[2])] + transform
        train_transform = transforms.Compose(transform)
        
        if size_scale_ratio is not None:
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        trainset = datasets.CIFAR10(root=path, train=True,
                                    download=True, transform=train_transform)
        test_trainset = datasets.CIFAR10(root=path, train=True,
                                    download=True, transform=test_transform)
        testset = datasets.CIFAR10(root=path, train=False,
                                   download=True, transform=test_transform)
        trainset = CustomDataset(trainset)
        test_trainset = CustomDataset(test_trainset)
        testset = CustomDataset(testset)

    elif (data_name == 'cifar100'):
        transform = [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if size_scale_ratio is not None:
            transform = [transforms.RandomResizedCrop(size_scale_ratio[0], scale=size_scale_ratio[1], ratio=size_scale_ratio[2])] + transform
        train_transform = transforms.Compose(transform)

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.CIFAR100(root=path, train=True,
                                    download=True, transform=train_transform)
        test_trainset = datasets.CIFAR100(root=path, train=True,
                                    download=True, transform=test_transform)
        testset = datasets.CIFAR100(root=path, train=False,
                                   download=True, transform=test_transform)
        
        trainset = CustomDataset(trainset)
        test_trainset = CustomDataset(test_trainset)
        testset = CustomDataset(testset)
    
    elif data_name == 'tiny_imagenet':
        
        # download tiny-imagenet
        if not os.path.exists(os.path.join(path, "tiny-imagenet")):
            source = os.path.join("/data/datasets", "tiny-imagenet.tar.gz")
            shutil.copy(source, path)

            # extracting tar file
            with tarfile.open(os.path.join(path, "tiny-imagenet.tar.gz"), "r:gz") as tar:
                tar.extractall(path=path)
        
        data_dir = os.path.join(path, "tiny-imagenet")
        transform = [transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        if size_scale_ratio is not None:
            transform = [transforms.RandomResizedCrop(size_scale_ratio[0], scale=size_scale_ratio[1], ratio=size_scale_ratio[2])] + transform
        train_transform = transforms.Compose(transform)

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        trainset = datasets.ImageFolder(os.path.join(path, data_dir, "train"), train_transform)
        test_trainset = datasets.ImageFolder(os.path.join(path, data_dir, "train"), test_transform)
        testset = datasets.ImageFolder(os.path.join(path, data_dir, "val"), test_transform)
        
        trainset = CustomDataset(trainset)
        test_trainset = CustomDataset(test_trainset)
        testset = CustomDataset(testset)
        
    return trainset, testset, test_trainset

def split_class_data(dataset, forget_class, num_forget):
    forget_index = []
    class_remain_index = []
    remain_index = []
    forget_count = {fc: 0 for fc in forget_class}
    
    targets_np = np.array(dataset.targets)
    for fc in forget_class:
        forget_index.extend(np.where(targets_np == fc)[0])
    remain_index = list(range(len(targets_np)))
    remain_index = list(set(remain_index) - set(forget_index))

    return forget_index, remain_index, class_remain_index

def get_unlearn_loader(trainset, testset, test_trainset, forget_class, batch_size, num_forget, repair_num_ratio=0.01):
    train_forget_index, train_remain_index, _ = split_class_data(trainset, forget_class,
                                                                                  num_forget=num_forget)
    test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class, num_forget=len(testset))
    test_train_forget_index, test_train_remain_index, _ = split_class_data(test_trainset, forget_class, num_forget=num_forget)

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  

    test_train_forget_sampler = SubsetRandomSampler(test_train_forget_index)  
    test_train_remain_sampler = SubsetRandomSampler(test_train_remain_index)  

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler)

    test_train_forget_loader = torch.utils.data.DataLoader(dataset=test_trainset, batch_size=batch_size,
                                                        sampler=test_train_forget_sampler)
    test_train_remain_loader = torch.utils.data.DataLoader(dataset=test_trainset, batch_size=batch_size,
                                                        sampler=test_train_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, test_train_forget_loader, test_train_remain_loader

def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    count = 0
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader
