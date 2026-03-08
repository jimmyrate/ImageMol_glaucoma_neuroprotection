import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd

from dataloader.image_dataloader import ImageDataset, load_filenames_multitask, get_datasets
from model.cnn_model_utils import load_model
from utils.public_utils import setup_device
from model.train_utils import load_smiles


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained ImageMol model')

    parser.add_argument('--dataset', type=str, default="BBBP", help='dataset name')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs')
    parser.add_argument('--workers', default=2, type=int, help='num workers')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--resume', type=str, required=True, help='path to trained checkpoint')
    parser.add_argument('--imageSize', type=int, default=224)
    parser.add_argument('--image_model', type=str, default="ResNet18")
    parser.add_argument('--output_csv', type=str, default="pred_results.csv", help="output csv file")

    return parser.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device, device_ids = setup_device(args.ngpu)

    # load dataset (只需要 test 全集即可)
    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
    names = load_filenames_multitask(args.image_folder, args.txt_file)
    names = np.array(names)

    img_transformer = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    labels = [0] * len(names)

    dataset = ImageDataset(names, labels, img_transformer=transforms.Compose(img_transformer),
                           normalize=normalize, args=args)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # load model
    model = load_model(args.image_model, imageSize=args.imageSize, num_classes=1)
    checkpoint = torch.load(args.resume, map_location=device)

    # 兼容不同保存方式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # 直接就是 state_dict

    # 加载参数（如果 key 不完全匹配，用 strict=False）
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # inference
    all_preds, all_names = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs.tolist())

    input_file = args.txt_file  # get_datasets 返回的第二个是 txt/csv 文件路径
    df = pd.read_csv(input_file, sep="\t")  # 如果是csv改成 sep="," 或者直接用 read_csv

    # 保证长度匹配
    assert len(df) == len(all_preds), f"数据行数 {len(df)} 和预测数 {len(all_preds)} 不一致！"

    # 添加预测列
    df["pred_score"] = all_preds

    # 保存新文件
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved predictions with scores to {args.output_csv}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
