from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve,auc, f1_score,cohen_kappa_score,mean_squared_error, mean_absolute_error, r2_score

from dataloader_ren import ImageMol_data
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import torchvision
from PIL import Image
import torch
import random


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png

        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================

    '''
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
    img.save(path)

def SavePairImage(smisA, smisB, path):
    mol1 = Chem.MolFromSmiles(smisA) 
    mol2 = Chem.MolFromSmiles(smisB) 
    # Chem.AllChem.Compute2DCoords(mol1)
    # Chem.AllChem.Compute2DCoords(mol2)

    img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2, subImgSize=(300, 300))
    img.save(path)

def SavePairImagePretrain(smisA, smisB, da, db, path='datasets'):
    mol1 = Chem.MolFromSmiles(smisA) 
    mol2 = Chem.MolFromSmiles(smisB) 
    # Chem.AllChem.Compute2DCoords(mol1)
    # Chem.AllChem.Compute2DCoords(mol2)

    img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2, subImgSize=(224, 224), margins=(5, 5))
    img.save(f'{path}/{da}_{db}_1.png')
    img = Draw.MolsToGridImage([mol2, mol1], molsPerRow=2, subImgSize=(224, 224))
    img.save(f'{path}/{da}_{db}_2.png')
    img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=1, molsPerCol=2, subImgSize=(224, 224))
    img.save(f'{path}/{da}_{db}_3.png')
    img = Draw.MolsToGridImage([mol2, mol1], molsPerRow=1, molsPerCol=2, subImgSize=(224, 224))
    img.save(f'{path}/{da}_{db}_4.png')

def SavePairImagePretrainV2(smisA, smisB, da, db, path='datasets'):
    mol1 = Chem.MolFromSmiles(smisA) 
    mol2 = Chem.MolFromSmiles(smisB) 
    # Chem.AllChem.Compute2DCoords(mol1)
    # Chem.AllChem.Compute2DCoords(mol2)
    mol1_img = Draw.MolToImage(mol1, size=(224, 224))
    mol2_img = Draw.MolToImage(mol2, size=(224, 224))
    row1 = Image.new('RGB', size=(224+224, 224))
    row2 = Image.new('RGB', size=(224+224, 224))
    col1 = Image.new('RGB', size=(224, 224+224))
    col2 = Image.new('RGB', size=(224, 224+224))

    row1.paste(mol1_img, (0,0))
    row1.paste(mol2_img, (224,0))
    row1.save(f'{path}/{da}_{db}_1.png')

    row2.paste(mol2_img, (0,0))
    row2.paste(mol1_img, (224,0))
    row2.save(f'{path}/{da}_{db}_2.png')

    col1.paste(mol1_img, (0,0))
    col1.paste(mol2_img, (0, 224))
    col1.save(f'{path}/{da}_{db}_3.png')

    col2.paste(mol2_img, (0,0))
    col2.paste(mol1_img, (0, 224))
    col2.save(f'{path}/{da}_{db}_4.png')


def evaluate_binary_class(y_pred, labels):
    # 将概率值转化为二进制值
    y_pred = np.argmax(y_pred, axis=1)
    # y_pred = [1 if i > 0.5 else 0 for i in y_pred]

    f1 = f1_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    roc_auc = roc_auc_score(labels, y_pred)

    return f1, recall, precision, acc, roc_auc

def evaluate_regression(y_pred, labels):
    # 将概率值转化为二进制值
    # y_pred = np.argmax(y_pred, axis=1)
    # y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    y_pred = y_pred.ravel()
    mse = mean_squared_error(labels, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, y_pred)
    r2 = r2_score(labels, y_pred)

    return mse, rmse, mae, r2


def evaluate_multi_class(y_pred, labels):
    # import ipdb; ipdb.set_trace()
    # roc_auc = roc_auc_score(labels, y_pred)
    
    # y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    scores = np.max(y_pred, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(labels, y_pred, average='macro')
    # aupr = f1_score(labels, y_pred, average='micro')
    # kappa = cohen_kappa_score(labels, y_pred)
    recall = recall_score(labels, y_pred, average='macro')
    precision = precision_score(labels, y_pred, average='macro')
    acc = accuracy_score(labels, y_pred)
    return f1, recall, precision, acc

def eval_loader(model, loader, device):
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    pre_loss = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        h, label = batch
        h = h.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(h)
            # pred = torch.softmax(pred, dim=1)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return evaluate_regression(np.array(Y_pre), np.array(Y_true))

def eval_mae_loader(model, loader, device):
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    pre_loss = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        img1, img2, label = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(img1, img2)
            pred = torch.softmax(pred, dim=1)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return evaluate_multi_class(np.array(Y_pre), np.array(Y_true))


def load_imagemol_data(args, fold=0):
    loaders = []
    for data_type in ['training', 'validation', 'test']:
        data = ImageMol_data(args.dataset, fold=fold, data_type=data_type)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        loaders.append(loader)
    # pickle.dump([loaders, data], open(path, 'wb'))
    return loaders

def save_model(args, model):
    path = '/root/autodl-tmp/ImageMol/ckpts/model_regression.pt'

    torch.save(model.state_dict(), path)

def load_model(args, model):
    path = '/root/autodl-tmp/ImageMol/ckpts/model_regression.pt'
    model.load_state_dict(torch.load(path))
    return model

def load_imagemol(modelname="ResNet18"):
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    
    # checkpoint = torch.load('/root/autodl-tmp/ImageMol/logs/glaucoma/valid_best_1213.pth')
    checkpoint = torch.load('/root/autodl-tmp/ImageMol/ckpts/pretraining-toy/checkpoints/ImageMol.pth.tar')
    # checkpoint = torch.load('ckpts/imagemol/CGIP.pth')
    ckp_keys = list(checkpoint['state_dict'])
    cur_keys = list(model.state_dict())
    model_sd = model.state_dict()
    
    ckp_keys = ckp_keys[:120]
    cur_keys = cur_keys[:120]

    for ckp_key, cur_key in zip(ckp_keys, cur_keys):
        model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

    model.load_state_dict(model_sd)
    arch = checkpoint['arch']
    print("resume model info: arch: {}".format(arch))

    # image_encoder = nn.Sequential(*list(model.children())[:-1])
    
    return model

def load_pretrained_component(pretrained_pth, model_key, consistency=False, logger=None,modelname = "ResNet18"):
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    # log = logger if logger is not None else logging
    flag = False  # load successfully when only flag is true
    desc = None
    if pretrained_pth:
        if os.path.isfile(pretrained_pth):
            # log.info("===> Loading checkpoint '{}'".format(pretrained_pth))
            checkpoint = torch.load(pretrained_pth)

            # load parameters
            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                model.load_state_dict(ckpt_model_state_dict)
                # log.info("load all the parameters of pre-trianed model.")
            else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(model.state_dict())
                len_ckp_keys = len(ckp_keys)
                len_cur_keys = len(cur_keys)
                model_sd = model.state_dict()

                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]

                # for idx in range(min(len_ckp_keys, len_cur_keys)):
                #     ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                #     # print(ckp_key, cur_key)
                #     model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                # model.load_state_dict(model_sd)
                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = ckpt_model_state_dict[ckp_key]

                model.load_state_dict(model_sd)
                # log.info("load the first {} parameters. layer number: model({}), pretrain({})"
                        #  .format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))
        # else:
        #     log.info("===> No checkpoint found at '{}'".format(pretrained_pth))
    # else:
    #     log.info('===> No pre-trained model')
    return model


if __name__ == '__main__':
    SavePairImagePretrainV2('CN1C2=C(C=C(Cl)C=C2)C(=NC(O)C1=O)C1=CC=CC=C1Cl', 'CC(C)(OC1=CC=C(C=C1)C(=O)C1=CC=C(Cl)C=C1)C(O)=O', 'a','a')