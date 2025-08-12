import os
import time
import random

import torch
from torch import nn

import argparse
import numpy as np
import tqdm

from utils import get_dataset, get_unlearn_loader, create_dir
from models import AllCNN, load_vit
from evaluation import all_eval, evaluate_KR
from mia import evaluate_mia



def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def exp_summary(args):
    print('*' * 100)
    print(' ' * 20 + 'Experiment Summary')
    print('*' * 100)
    print(f"Experiment Name: {args.exp}")
    print(f"Method: {args.method}")
    if args.method == 'ESC':
        print(f"Pruning Hyperparameter (p): {args.p}%")
    elif args.method == 'ESC_T':
        print(f"Threshold for ESC-T: {args.threshold}")
    print(f"Data Name: {args.data_name}")
    print(f"Forget Class: {args.forget_class}")
    print(f"Model Name: {args.model_name}")
    print('*' * 100)


def prepare_dataset(args):
    '''
    Prepare dataset and dataloaders for unlearning
    Notation:
    - trfl: forgetting training dataloader
    - trrl: remaining training dataloader
    - tefl: forgetting testing dataloader
    - terl: remaining testing dataloader
    - ttfl: forgetting training dataloader for testing
    - ttrl: remaining training dataloader for testing
    '''
    # Dataset 
    if args.model_name == 'vit_base_patch16_224':
        input_size = 224
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        size_scale_ratio = [input_size, scale, ratio]
    else:
        size_scale_ratio = None
    trainset, testset, test_trainset = get_dataset(args.data_name, args.dataset_dir, size_scale_ratio)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    # Forget & Remain Set (number of samples to a single class)
    if args.data_name in ['cifar100', 'tiny_imagenet']:
        num_forget = 500 
    else:
        num_forget = 5000

    # Unlearn Dataloader
    trfl, _, tefl, terl, ttfl, ttrl \
        = get_unlearn_loader(trainset, testset, test_trainset, args.forget_class, args.batch_size, num_forget)

    num_classes = max(trainset.targets) + 1

    return trfl, tefl, terl, ttfl, ttrl, test_loader, train_loader, num_classes


def parse_args():
    parser = argparse.ArgumentParser("ESC")
    parser.add_argument('--exp', type=str, default='ESC_cifar10', help='experiment name')
    parser.add_argument('--method', type=str, default='ESC', choices=['ESC', 'ESC_T'], help='ESC unlearning method')

    ####### Data setting #######
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='dataset, among [cifar10, cifar100, tiny_imagenet]')
    parser.add_argument('--dataset_dir', type=str, default='/local_datasets', help='dataset directory')
    parser.add_argument('--forget_class', nargs='+', type=int, default=[4], help='List of the forgetting classes, for reproduce using *4 index')

    ####### Model setting #######
    parser.add_argument('--model_name', type=str, default='AllCNN', choices=['AllCNN', 'resnet_18', 'vit_base_patch16_224'], help='select the model name')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='checkpoints directory')
    
    ####### Experimental setting #######
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    
    ########Evaluation setting#######
    parser.add_argument('--evaluation', action='store_true', help='evaluate utility of unlearn model')
    parser.add_argument('--mia', action='store_true', help='evaluate mia of unlearn model')
    parser.add_argument('--use_pytorch_mia', action='store_true', help='Use PyTorch-based MIA instead of Logistic Regression')
    parser.add_argument('--mia_batch_size', type=int, default=32, help='batch size for MIA')
    parser.add_argument('--mia_lr', type=float, default=1e-4, help='learning rate for MIA')
    parser.add_argument('--kr', action='store_true', help='evaluate Knowledge Retention (KR) of unlearn model')
    parser.add_argument('--kr_lp', type=float, default=1e-3, help='learning rate for Knowledge Retention')
    parser.add_argument('--kr_epoch', type=int, default=10, help='epoch for Knowledge Retention')
    parser.add_argument('--kr_batch_size', type=int, default=64, help='batch size for Knowledge Retention')
    
    ####### ESC(-T) setting #######
    parser.add_argument('--p', type=float, default=1.5, help='pruning hyperparameter for ESC')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold for ESC-T')

    args = parser.parse_args()

    return args


def main(args):
    # Set seed
    seed_torch(seed=args.seed)
    
    # Summary for experiment
    exp_summary(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create directories
    exp_dir = f"experiments/{args.exp}"
    ckpt_dir = f"experiments/{args.exp}/checkpoints/"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for arg in vars(args).items():
            f.write(f"{arg}\n")

    create_dir(args.dataset_dir)
    create_dir(args.checkpoint_dir)
    path = args.checkpoint_dir + '/'

    # Dataset 
    trfl, tefl, terl, ttfl, ttrl, test_loader, train_loader, num_classes = prepare_dataset(args)
    
    if args.model_name == 'AllCNN':
        model = AllCNN(n_channels=3, num_classes=num_classes, filters_percentage=0.5)
        state = torch.load('{}.pth'.format(path + args.data_name + "_ori_allcnn"),)

    elif args.model_name == 'vit_base_patch16_224':
        model = load_vit(args.model_name, num_classes=num_classes, device=device, is_pretrained=False, is_backbone_freezed=True)
        state = torch.load('{}.pth'.format(path + args.data_name + "_ori_vit"),).state_dict()

    else:
        raise NotImplementedError(f"Model {args.model_name} is not implemented.")
    
    model.load_state_dict(state)
    model.to(device)
    del state
    

    # Start unlearning
    if args.method == "ESC":
        print('*' * 100)
        print(' ' * 20 + 'begin ESC unlearning')
        print('*' * 100)

        start = time.time()

        # save embedding features
        data_len = len(trfl.dataset)
        if args.model_name == 'AllCNN':
            feat_log = torch.zeros(data_len, int(192 * 0.5))
        elif args.model_name == 'resnet_18':
            feat_log = torch.zeros(data_len, 512)       
        elif args.model_name == 'vit_base_patch16_224':
            feat_log = torch.zeros(data_len, 768)
        else:
            raise NotImplementedError(f"Model {args.model_name} is not implemented.")

        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm.tqdm(trfl)):
                x = x.to(device, non_blocking=True)
                start_ind = i * args.batch_size
                end_ind = min((i + 1) * args.batch_size, data_len)
                output = model(x, all=True)

                if args.batch_size == output['pre_logits'].shape[0]:
                    feat_log[start_ind:end_ind, :] = output['pre_logits']
                else:
                    end_ind = i * args.batch_size + output['pre_logits'].shape[0]
                    feat_log[start_ind:end_ind, :] = output['pre_logits']

        # singular value decomposition
        u, _, _ = torch.svd(feat_log.T.to(device))

        # only use bottom p% singular vectors
        if args.model_name == 'AllCNN':
            prune_k = int(192 * 0.5 * args.p / 100)
        elif args.model_name == 'resnet_18':
            prune_k = int(512 * args.p / 100)
        elif args.model_name == 'vit_base_patch16_224':
            prune_k = int(768 * args.p / 100)
        else:
            raise NotImplementedError(f"Model {args.model_name} is not implemented.")
        u_p = u[:, prune_k:]
        
        model.esc_set(u_p)

        end = time.time()
        print('ESC unlearning time:', end-start, 's')

        # save model
        torch.save(model, '{}.pth'.format(ckpt_dir + "ESC_unlearned_model"))

    elif args.method == "ESC_T":
        print('*' * 100)
        print(' ' * 20 + 'begin ESC_T unlearning')
        print('*' * 100)

        start = time.time()

        # save embedding features
        data_len = len(trfl.dataset)
        print(data_len)
        if args.model_name == 'AllCNN':
            feat_log = torch.zeros(data_len, int(192 * 0.5))
        elif args.model_name == 'resnet_18':
            feat_log = torch.zeros(data_len, 512)       
        elif args.model_name == 'vit_base_patch16_224':
            feat_log = torch.zeros(data_len, 768)
        else:
            raise NotImplementedError(f"Model {args.model_name} is not implemented.")

        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm.tqdm(trfl)):
                x = x.to(device, non_blocking=True)
                start_ind = i * args.batch_size
                end_ind = min((i + 1) * args.batch_size, data_len)
                output = model(x, all=True)

                if args.batch_size == output['pre_logits'].shape[0]:
                    feat_log[start_ind:end_ind, :] = output['pre_logits']
                else:
                    end_ind = i * args.batch_size + output['pre_logits'].shape[0]
                    feat_log[start_ind:end_ind, :] = output['pre_logits']

        # singular value decomposition
        u, _, _ = torch.svd(feat_log.T.to(device))
        
        mask = torch.ones_like(u)
        
        criterion = nn.CrossEntropyLoss()

        for epo in tqdm.tqdm(range(args.epoch)):
            for x, y in trfl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                mask = mask.detach() 
                mask.requires_grad_(True)

                model.esc_set(u * mask, esc_t=True)
                outputs = model(x)

                pred = outputs.argmax(dim=1)
                learned = (y == pred)

                if learned.any():
                    loss = -criterion(outputs[learned], y[learned])
                    loss.backward()

                    if mask.grad is not None:
                        with torch.no_grad():
                            mask = mask - args.lr * mask.grad
                            mask = torch.clamp(mask, min=0, max=1)
                    mask.grad = None

            model.esc_set(u * mask, esc_t=True)
        
            model.eval()
            with torch.no_grad():
                num_hits = 0
                for i, (x, y) in (enumerate(trfl)):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                    outputs = model(x)
                    pred = outputs.argmax(dim=1)
                    num_hits += (y == pred).sum().item()
            if num_hits == 0:
                break

        mask = (mask > args.threshold).to(mask.dtype)

        model.esc_set(u * mask, esc_t=True)

        end = time.time()
        print('ESC-T unlearning time:', end-start, 's')

        # save model
        torch.save(model, '{}.pth'.format(ckpt_dir + "ESC_T_unlearned_model"))

    if args.evaluation:
        with torch.no_grad():
            all_eval(model, test_loader, ttfl, ttrl, tefl, terl, device)

    if args.mia:
        evaluate_mia(model, trfl, tefl, device, args)

    if args.kr:
        evaluate_KR(model, train_loader, test_loader, ttfl, ttrl, tefl, terl, num_classes, ckpt_dir=ckpt_dir, device=device, args=args)

    return model


if __name__ == '__main__':
    args = parse_args()
    main(args)