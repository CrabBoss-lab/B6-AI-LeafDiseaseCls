# -*- codeing = utf-8 -*-
# @Time :2023/4/11 20:02
# @Author :yujunyu
# @Site :
# @File :predict1.py
# @software: PyCharm
import matplotlib
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time

from torchvision.models import resnet18
from net import Net


class Predict():
    def __init__(self, model):
        super(Predict, self).__init__()
        self.model = model
        self.CUDA = torch.cuda.is_available()
        # self.net = resnet18(pretrained=False, num_classes=3)
        self.net = Net()
        if self.CUDA:
            self.net.cuda()
            device = 'cpu'
        else:
            device = 'cpu'
        state = torch.load(self.model, map_location=device)
        self.net.load_state_dict(state)
        print('模型加载完成！')
        self.net.eval()

    @torch.no_grad()
    def recognize(self, img):
        with torch.no_grad():
            if self.CUDA:
                img = img.cuda()
            img = img.view(-1, 3, 224, 224)  # 等于reshape
            y = self.net(img)
            p_y = torch.nn.functional.softmax(y, dim=1)
            # print(p_y)
            p, cls_index = torch.max(p_y, dim=1)
            return cls_index.cpu(), p.cpu()


if __name__ == '__main__':
    # 模型权重
    model = 'predict_demo/weight.pth'
    recognizer = Predict(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4737, 0.4948, 0.4336], std=[0.1920, 0.1592, 0.2184])

    ])

    classes = ['番茄叶斑病', '苹果黑星病', '葡萄黑腐病']
    # label_map: {'番茄叶斑病': 0, '苹果黑星病': 1, '葡萄黑腐病': 2}

    # 预测单张图片
    # img_file = 'dataset/test/番茄叶斑病/番茄叶斑病 (25).JPG'
    # print(f'预测单张图片:{img_file}')
    # img = Image.open(img_file)
    # img2 = Image.fromarray(np.uint8(img))
    # img2= transform(img2)
    # # img = torch.reshape(img, (1, 3, 32, 32))
    # st = time.time()
    # cls, p = recognizer.recognize(img2)
    # print(f'推理时间:{time.time() - st}')
    # text='预测类别:{}\n预测概率:{:}'.format(classes[cls], p.numpy()[0])
    # print(text)
    # # matplotlib可视化预测图片
    # plt.imshow(img)
    # # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title(text)
    # plt.show()


    # # 预测多张图片
    folder_path = 'dataset/test'
    files = os.listdir(folder_path)
    # 得到每个img文件地址
    images_files = [os.path.join(folder_path, f) for f in files]
    #
    sum_true=0
    sum_all=0
    for img in images_files:
        true_label = img.split('\\')[-1].split('.')[0]
        # print(img)
        imgs = os.listdir(img)
        img_path = [os.path.join(img, f) for f in imgs]
        for img_path in img_path:
            # print(img_path)
            # print(img,true_label)
            image = Image.open(img_path)
            image = Image.fromarray(np.uint8(image))
            image = transform(image)
            st = time.time()
            cls, p = recognizer.recognize(image)
            if true_label == classes[cls]:
                sum_true += 1
            print(f'推理时间:{time.time() - st}\t真实标签:{true_label}\t 预测标签:{classes[cls]}\t 预测概率:{p.numpy()[0]}')
            sum_all += 1
    print(f'总数量:{sum_all}')
    print(f'正确数:{sum_true}')
    print(f'准确率:{sum_true / sum_all}')
