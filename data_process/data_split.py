import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
file_path = '/root/autodl-tmp/ImageMol/datasets/antibiotics/data_for_regression/antibiotic_inhibition_for_train.csv'
data = pd.read_csv(file_path)

# 划分数据集，训练集占80%，临时集占20%
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

# 划分临时集为测试集和验证集，各占50%
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存划分后的数据集为新的CSV文件
train_data.to_csv('/root/autodl-tmp/ImageMol/datasets/antibiotics/data_for_regression/antibiotic_inhibition_train.csv', index=False)
test_data.to_csv('/root/autodl-tmp/ImageMol/datasets/antibiotics/data_for_regression/antibiotic_inhibition_test.csv', index=False)
val_data.to_csv('/root/autodl-tmp/ImageMol/datasets/antibiotics/data_for_regression/antibiotic_inhibition_val.csv', index=False)

print(f'Train set size: {len(train_data)}')
print(f'Test set size: {len(test_data)}')
print(f'Validation set size: {len(val_data)}')
