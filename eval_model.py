import os
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from model_encoder import ImageMol, ImageMol_Regression
from torch.utils.data import DataLoader

class ImageMol_data(Dataset):
    def __init__(self, path):
        super().__init__()
        self.img_transformer_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

        self.samples = []
        h = path
        self.samples.append(h)

    def __image_process(self, img):
        return self.img_transformer_test(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        h = self.samples[index]
        img_h = Image.open(h)
        img_h = self.__image_process(img_h)
        return img_h

def load_model(model):
    path = '/root/autodl-tmp/ImageMol/ckpts/model_regression.pt'
    model.load_state_dict(torch.load(path))
    return model


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

def smiles2img(smile,idx):
    '''
    demo of raw_file_path:
        index,k_100,k_1000,k_10000,smiles
        1,97,861,4925,CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1
        2,37,524,4175,CC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1
        3,77,636,6543,COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC
        ...
    :return:
    '''
    save_path = '/root/autodl-tmp/ImageMol/datasets/antibiotics/test/'
    img_save_root = os.path.join(save_path, "224")

    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)
    mol = Chem.MolFromSmiles(smile)
    if mol:
        filename = "{}.png".format(idx)
        img_save_path = os.path.join(img_save_root, filename)


        try:
            loadSmilesAndSave(smile, img_save_path)
        except:
            pass
        return img_save_path
    else:
        return None


if __name__ == '__main__':
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImageMol_Regression()
    model = load_model(model)
    model = model.to(device)
    data = pd.read_csv('/root/autodl-tmp/ImageMol/datasets/antibiotics/test/test_chemprop0716.csv')
    smiles = data['SMILES'].values
    pred_all = []
    for idx, smile in enumerate(tqdm(smiles)):
        smile_path = smiles2img(smile, idx)
        if smile_path:
            smiles_data = ImageMol_data(smile_path)
            data_loder = DataLoader(smiles_data, batch_size=1, shuffle=False)
            model.eval()
            for batch in data_loder:
                smils_img = batch
                smils_img = smils_img.to(device)
                with torch.no_grad():
                    pred = model(smils_img)
                    pred = pred.item()
                    pred_all.append(pred)
                if os.path.exists(smile_path):
                    os.remove(smile_path)
        else:
            pred_all.append(None)

    if len(pred_all) == len(data):
        data['predict'] = pred_all
        data.to_csv('/root/autodl-tmp/ImageMol/datasets/antibiotics/test/test_chemprop0716_result.csv', index=False)


