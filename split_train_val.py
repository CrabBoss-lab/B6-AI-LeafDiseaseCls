import os
import random
import shutil

# 设置随机数种子
random.seed(42)

# 数据集路径
data_path = r'C:\Users\yujunyu\Desktop\data'

# 类别标签
labels = os.listdir(data_path)

# 训练集和验证集比例
train_ratio = 0.8

# 创建训练集和验证集文件夹
train_dir = 'dataset/train'
val_dir = 'dataset/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别的文件夹
for label in labels:
    # 获取当前类别的图片文件名列表
    img_list = os.listdir(os.path.join(data_path, label))
    # 打乱图片文件名列表
    random.shuffle(img_list)
    # 计算训练集和验证集的分割点
    split_point = int(len(img_list) * train_ratio)
    # 将前split_point个文件复制到训练集文件夹
    for img_name in img_list[:split_point]:
        src_path = os.path.join(data_path, label, img_name)
        dst_path = os.path.join(train_dir, label, img_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    # 将后面的文件复制到验证集文件夹
    for img_name in img_list[split_point:]:
        src_path = os.path.join(data_path, label, img_name)
        dst_path = os.path.join(val_dir, label, img_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
