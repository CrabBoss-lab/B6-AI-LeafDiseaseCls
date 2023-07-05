import os

# 数据集路径
data_path = 'dataset'
print('数据集路径:', data_path)

# 类别标签
labels = os.listdir(data_path)
# 统计每个文件夹下图像的数量
for label in labels:
    img_list = os.listdir(os.path.join(data_path, label))
    print(f"{label}/:{len(img_list)}类")
    sum = 0
    for i in img_list:
        try:
            i_list = os.listdir(os.path.join(data_path, label, i))
            sum += len(i_list)
            print(f"\t-{i}: {len(i_list)}张")
        except:
            pass
    print(f"{label}/: 共{sum}张")
    print('-' * 20)

