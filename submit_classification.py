# -*- coding: UTF-8 -*-
import argparse
import os
import sys
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import numpy as np
# import timm
import torch.nn as nn
import pandas as pd
import torchvision
import torch.nn.functional as F

from utils.public_utils import setup_device, cal_torch_model_params, is_left_better_right
from model.train_utils import fix_train_random_seed
from dataloader.image_dataloader import ImageDataset


def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    return model


def main(args):

    device, device_ids = setup_device(1)
    
    # fix random seeds
    fix_train_random_seed(args.runseed)

    ##################################### load data #####################################

    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    data = pd.read_csv("/home/yanqipeng/RXB/ImageMol/datasets/glaucoma_candidate/raw/glaucoma_candidates.csv")
    
    test_smi = data['SMILES'].tolist()
    test_label = [0] * len(test_smi)

    test_dataset = ImageDataset(test_smi, test_label, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### load model #####################################
    model = load_model(num_classes=2)

    # resume = "./ckpts/finetuning/" + args.resume + ".pth"
    resume = "/home/yanqipeng/RXB/ImageMol/logs/glaucoma/valid_best.pth"

    if resume:
        if os.path.isfile(resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loading completed")
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    model = model.cuda()

    ##################################### train #####################################
    model.eval()

    y_prob = []
    data_loader = tqdm(test_dataloader)
    for step, data_ in enumerate(data_loader):
        images, _ = data_
        images = images.to(device)
        
        with torch.no_grad():
            pred = model(images)
            
            pred = F.softmax(pred, dim=1)
        
        y_prob.extend(pred[:,1].cpu().numpy().reshape(-1).tolist())
    

    data = {}

    data['smiles'] = test_smi
    data['pred'] = y_prob
    data = pd.DataFrame(data)
    data.to_csv('/home/yanqipeng/RXB/ImageMol/datasets/glaucoma_candidate/raw/glaucoma_result.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')

    # basic
    parser.add_argument('--dataset', type=str, default="EP4", help='dataset name')
    parser.add_argument('--dataroot', type=str, default="./datasets/finetuning/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=5, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--batch', default=16, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--resume', default='ImageMol_EP4', type=str, metavar='PATH',
                        help='path to checkpoint 85798656 (default: None) ./ckpts/pretrain/checkpoints/MMViT_1.pth.tar')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')

    args = parser.parse_args()
    main(args)
