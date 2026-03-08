import os
import torch
import argparse
from torch.optim import Adam,AdamW
from tqdm import tqdm
import numpy as np
from model_encoder import ImageMol, ImageMol_Regression,ImageMol_Classification
import torch.nn.functional as F
from layers import Regularization
# from utils import eval_loader, save_model, load_model, load_imagemol_data
from utils_REN import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(os.path.join('logs', 'image_ddi_2080ti', 'imagemol'))

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='dropout probability')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='epoch')
    parser.add_argument('--fold', type=int, default=0,
                        help='[0, 1, 2, 3, 4, 5] 5 for demo data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')


    args = parser.parse_args()
    # if args.dataset == 'Deng':
    #     args.num_classes = 65
    # elif args.dataset == 'Ryu' or args.dataset == 'drugbank_ind':
    #     args.num_classes = 86
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args

def main(args):
    # model = ImageMol_Regression()
    model = ImageMol_Classification()
    model = model.to(args.device)

    loaders = load_imagemol_data(args, fold=0)
    train, valid, test = loaders

    #class
    loss_func = F.cross_entropy

    # criterion = nn.MSELoss()

    # reg = Regularization(model, args.weight_decay)
    # reg = reg.to(args.device)

    optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-4, weight_decay=args.weight_decay)
    early_stop = 0
    best_acc = -1.
    # best_mse = float('inf')
    for i in range(args.epoch):
        bar = tqdm(enumerate(train))
        loss_t = 0.
        for idx, batch in bar:

            img_h, label = batch
            label = label.long()
            # label = label.float()
            img_h = img_h.to(args.device)
            label = label.to(args.device)

            optim.zero_grad()
            logits= model(img_h)
            # import ipdb;ipdb.set_trace();
            # loss = criterion(logits,label)
            loss = loss_func(logits, label)
            # loss += reg(model)
            loss.backward()
            optim.step()

            loss_t+=loss.detach().cpu().item()
            bar.set_description('Training: epoch-{} |'.format(i+1) + str(idx+1) + '/{} loss_train: '.format(len(train)) + str(loss.cpu().detach().numpy()))

        loss_print = loss_t/(idx+1)
        early_stop+=1


        (f1, recall, precision, acc,roc_auc) = eval_loader(model, valid, args.device)
        # (mse, rmse, mae, r2) = eval_loader(model, valid, args.device)
        writer.add_scalar('cross_loss', loss_print, global_step=i)
        writer.add_scalar('f1', f1, global_step=i)
        writer.add_scalar('recall', recall, global_step=i)
        writer.add_scalar('precision', precision, global_step=i)
        writer.add_scalar('acc', acc, global_step=i)
        writer.add_scalar('roc_auc', roc_auc, global_step=i)

        # writer.add_scalar('cross_loss', loss_print, global_step=i)
        # writer.add_scalar('mse', mse, global_step=i)
        # writer.add_scalar('rmse', rmse, global_step=i)
        # writer.add_scalar('mae', mae, global_step=i)
        # writer.add_scalar('r2', r2, global_step=i)

        # print('Epoch: {} | train_loss: {},  mse: {}, rmse: {}, mae: {},r2: {}'.format(i + 1,
        #                                                                                 loss_print,
        #                                                                                 mse,
        #                                                                                 rmse,
        #                                                                                 mae, r2))
        print('Epoch: {} | train_loss: {}, f1: {}, recall: {}, precision: {}, acc: {},roc_auc: {}'.format(i+1, loss_print, f1, recall, precision, acc,roc_auc))
        # if f1 > best_acc:
        #     best_acc = f1
        #     save_model(args, model)
        #     print('best model saved!!!')
        #     # log_info(args, 'Best model saved, Epoch: {} | train_loss: {}, auc: {}, recall: {}, precision: {}, acc: {}'.format(i+1, loss_print, roc_auc, recall,precision, acc))
        #     early_stop = 0
        # if early_stop > 20:
        #     break
        # if mse < best_mse:
        #     best_mse = mse
        #     save_model(args, model)
        #     print('Best model saved!!!')
        #     # log_info(args, 'Best model saved, Epoch: {} | train_loss: {}, val_mse: {}'.format(i+1, train_loss, val_mse))
        #     early_stop = 0  # 重置early stopping计数器
        # if early_stop > 5:
        #     print('Early stopping...')
        #     break
        if f1 > best_acc:
            best_acc = f1
            save_model(args, model)  # 保存最佳模型
            print('Best model saved!!!')
            early_stop = 0  # 重置early stopping计数器

        if early_stop > 50:
            print('Early stopping...')
            break
    model = load_model(args, model)
    # (mse, rmse, mae, r2) = eval_loader(model, test, args.device)
    (f1, recall, precision, acc, roc_auc) = eval_loader(model, test, args.device)
    
    print('Test: f1: {}, recall: {}, precision: {}, acc: {},roc_auc: {}'.format(f1, recall, precision, acc,roc_auc))


if __name__ == '__main__':
    args = parse_params()
    main(args)