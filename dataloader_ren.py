import os
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class ImageMolDDI(Dataset):
    def __init__(self, data_name = 'Deng', fold=0, data_type='training'):
        super().__init__()
        if data_name == 'Deng':
            path = f'datasets/Deng_dataset/{fold}/ddi_{data_type}1.csv'
        elif data_name == 'Ryu':
            path = f'datasets/Ryu_dataset/{fold}/ddi_{data_type}1.csv'
        elif data_name == 'drugbank_ind':
            path = f'datasets/db_ind/{fold}/ddi_{data_type}1.csv'

        print(path)
        self.img_transformer_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

        self.drug_entityid = {}
        with open('ckpts/drkg_rotate/drug_entityid.csv', 'r') as f:
            for line in f:
                did, eid = line.strip().split('\t')
                self.drug_entityid[did] = int(eid)

        self.samples = []
        with open(path, 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                h, label, t = line.strip().split(',')
                # if h not in self.drug_entityid or t not in self.drug_entityid:
                #     continue
                # h_id = self.drug_entityid[h]
                # t_id = self.drug_entityid[t]
                h_id = 0
                t_id = 0
                h = f'datasets/drug_image/{h}.png'
                t = f'datasets/drug_image/{t}.png'
                label = int(label)
                self.samples.append([h, h_id, t, t_id, label])
        
        random.shuffle(self.samples)

    def __image_process(self, img):
        return self.img_transformer_test(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        h, h_id, t, t_id, label = self.samples[index]
        img_h = Image.open(h)
        img_t = Image.open(t)
        img_h = self.__image_process(img_h)
        img_t = self.__image_process(img_t)

        return img_h, h_id, img_t, t_id, label


class ImageMol_data(Dataset):
    def __init__(self, data_name='Deng', fold=0, data_type='training'):
        super().__init__()
        # if data_name == 'Deng':
        #     path = f'datasets/Deng_dataset/{fold}/ddi_{data_type}1.csv'
        # elif data_name == 'Ryu':
        #     path = f'datasets/Ryu_dataset/{fold}/ddi_{data_type}1.csv'
        # elif data_name == 'drugbank_ind':
        #     path = f'datasets/db_ind/{fold}/ddi_{data_type}1.csv'

        # print(path)
        if data_type == 'training':
            path = '/root/autodl-tmp/ImageMol/datasets/glaucoma/data/glaucoma_data_train.csv'
        elif data_type == 'validation':
            path = '/root/autodl-tmp/ImageMol/datasets/glaucoma/data/glaucoma_data_val.csv'
        elif data_type == 'test':
            path = '/root/autodl-tmp/ImageMol/datasets/glaucoma/data/glaucoma_data_test.csv'
        print(path)
        self.img_transformer_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

        # self.drug_entityid = {}
        # with open('ckpts/drkg_rotate/drug_entityid.csv', 'r') as f:
        #     for line in f:
        #         did, eid = line.strip().split('\t')
        #         self.drug_entityid[did] = int(eid)

        self.samples = []
        with open(path, 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                index, h, label= line.strip().split(',')
                h = f'/root/autodl-tmp/ImageMol/datasets/glaucoma/data/224/{h}'
                label = float(label)
                self.samples.append([h, label])

        random.shuffle(self.samples)

    def __image_process(self, img):
        return self.img_transformer_test(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        h, label = self.samples[index]
        img_h = Image.open(h)
        img_h = self.__image_process(img_h)

        return img_h, label


class SimpleBinaryDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        初始化数据加载器。
        参数:
        - csv_file: 包含特征和标签的 CSV 文件路径。
        - transform: 对特征进行预处理的函数（可选）。
        """
        # 读取 CSV 文件
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        根据索引获取一个样本，包括特征和标签。
        参数:
        - index: 要获取的样本的索引。
        返回:
        - feature: 特征数据，经过转换处理（如果有指定转换函数的话）。
        - label: 标签数据。
        """
        # 从 DataFrame 中获取特征和标签
        x = self.dataframe.iloc[index, 0]  # 假设特征在第一列
        y = self.dataframe.iloc[index, 1]  # 假设标签在第二列

        if self.transform:
            x = self.transform(x)

        return x, y


class PromptDDIDataset(Dataset):
    def __init__(self,image_process, data_name = 'Deng', fold=0, data_type='training'):
        super().__init__()
        if data_name == 'Deng':
            path = f'datasets/Deng_dataset/{fold}/ddi_{data_type}1.csv'
        elif data_name == 'Ryu':
            path = f'datasets/Ryu_dataset/{fold}/ddi_{data_type}1.csv'
        print(path)
        self.image_process = image_process

        drug_entityid = {}
        with open('datasets/Deng_dataset/drug_entityid.csv', 'r') as f:
            for line in f:
                did, eid = line.strip().split('\t')
                drug_entityid[did] = int(eid)

        self.samples = []
        with open(path, 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                h, label, t = line.strip().split(',')
                file_name = f'datasets/{data_name}_dataset/img/{h}_{t}.png'
                label = int(label)
                self.samples.append([file_name, label])
        
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, label = self.samples[index]
        img = Image.open(file_name)
        img = self.image_process(img)

        return img, label


if __name__ == '__main__':
    pass