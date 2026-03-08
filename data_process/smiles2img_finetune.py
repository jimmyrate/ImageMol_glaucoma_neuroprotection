# from rdkit.Chem import Draw
# import argparse
# import os
# import pandas as pd
# from tqdm import tqdm
# from rdkit import Chem
#
#
#
# # python data_process/smiles2img_finetune.py
#
# def main():
#     parser = argparse.ArgumentParser(description='csv2image')
#     # SMILES数据集所在的文件夹
#     parser.add_argument('--dataroot', type=str, default="datasets/finetuning/EP4/raw", help='data root')
#     parser.add_argument('--csvname', type=str, default="EP4_processed_ac.csv", help='The name of CSV,放在--dataroot目录下')
#     parser.add_argument('--outputImageDir', type=str, default="datasets/finetuning/EP4/processed/224")
#     parser.add_argument('--theCsvFirstLineSMILES', type=str, default="SMILES", help='CSV你要转换的列名')
#
#     args = parser.parse_args()
#     # SMILES数据集名称
#     raw_file_path = os.path.join(args.dataroot, args.csvname)
#     # 输出图像文件夹名字
#     opt_img = os.path.join(args.outputImageDir)
#
#     if not os.path.exists(opt_img):
#         os.makedirs(opt_img)
#
#     df = pd.read_csv(raw_file_path)
#
#     # 这是csv文件的第一行索引
#     opt_smiles = df[args.theCsvFirstLineSMILES].values
#
#     i = 1
#
#     for opt_smiles in tqdm(opt_smiles, total=len(opt_smiles)):
#         try:
#             filename = "{}.png".format(i)
#             i = i + 1
#             opt_img_save_path = os.path.join(opt_img, filename)
#
#             mol = Chem.MolFromSmiles(opt_smiles)
#             img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
#             img.save(opt_img_save_path)
#
#         except Exception as e:
#             print(e)
#             print("The current SMILES line is ", i, ", and the error SMILES is :", opt_smiles)
#
#     print("The total SMILES is :", i)
#
#
# if __name__ == '__main__':
#     main()
#
#
#

from rdkit.Chem import Draw
import argparse
import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem


# python data_process/smiles2img_finetune.py

def main():
    parser = argparse.ArgumentParser(description='csv2image')
    # SMILES数据集所在的文件夹
    parser.add_argument('--dataroot', type=str, default="/home/yanqipeng/RXB/ImageMol/datasets/finetuning/glaucoma/raw", help='data root')
    parser.add_argument('--csvname', type=str, default="/home/yanqipeng/RXB/ImageMol/datasets/finetuning/glaucoma/raw/glaucoma_data.csv", help='The name of CSV,放在--dataroot目录下')
    parser.add_argument('--outputImageDir', type=str, default="datasets/finetuning/EP4/processed/224")
    parser.add_argument('--theCsvFirstLineSMILES', type=str, default="SMILES", help='CSV你要转换的列名')
    parser.add_argument('--outputCsvName', type=str, default="EP4_cleaned.csv", help='Cleaned CSV file name')

    args = parser.parse_args()
    # SMILES数据集路径
    raw_file_path = os.path.join(args.dataroot, args.csvname)
    # 输出图像文件夹路径
    opt_img = os.path.join(args.outputImageDir)

    if not os.path.exists(opt_img):
        os.makedirs(opt_img)

    # 读取CSV文件
    df = pd.read_csv(raw_file_path)

    # 这是CSV文件的SMILES列
    opt_smiles_list = df[args.theCsvFirstLineSMILES].values

    cleaned_df = df.copy()  # 创建一个拷贝以存储清理后的数据
    i = 1
    rows_to_remove = []  # 记录无法转换的行索引

    for index, opt_smiles in tqdm(enumerate(opt_smiles_list), total=len(opt_smiles_list)):
        try:
            filename = "{}.png".format(i)
            i += 1
            opt_img_save_path = os.path.join(opt_img, filename)

            mol = Chem.MolFromSmiles(opt_smiles)
            if mol is None:  # 如果SMILES格式无效
                raise ValueError("Invalid SMILES")

            img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
            img.save(opt_img_save_path)

        except Exception as e:
            print(e)
            print("Error at index:", index, "SMILES:", opt_smiles)
            rows_to_remove.append(index)

    # 删除错误行
    cleaned_df = cleaned_df.drop(index=rows_to_remove)

    # 保存清理后的CSV
    # cleaned_csv_path = os.path.join(args.dataroot, args.outputCsvName)
    cleaned_df.to_csv(args.outputCsvName, index=False)

    print(f"The total processed SMILES is: {len(cleaned_df)}")
    # print(f"Cleaned CSV saved to: {cleaned_csv_path}")


if __name__ == '__main__':
    main()
