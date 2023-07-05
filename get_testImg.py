import os
import random
import shutil

# 设置随机种子
random.seed(40)
# 设置文件夹路径
original_folder = r'C:\Users\yujunyu\Desktop\data'
target_folder = './dataset/test'
# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 指定原始图片文件夹路径和目标文件夹路径
for item in os.listdir(original_folder):
    item_path = os.path.join(original_folder, item)
    print(item_path)
    # 获取原始文件夹下所有图片文件的路径
    image_files = [os.path.join(item_path, file) for file in os.listdir(item_path) if file.endswith(('JPG', 'jpg'))]
    # 随机选择20张图片
    random_images = random.sample(image_files, 20)
    print(random_images)
    # 将这些图片复制到目标文件夹中
    for image in random_images:
        shutil.copy(image, target_folder)
    # # 删除原始文件夹下的这些图片
    for image in random_images:
        os.remove(image)
