import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 26 * 26, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # 前向传播函数
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 26 * 26)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义预测类
class Predict:
    def __init__(self, model):
        self.model = model
        # 创建神经网络模型
        self.net = Net()
        # 加载预训练模型参数
        state = torch.load(self.model, map_location='cpu')
        self.net.load_state_dict(state)
        print('模型加载完成！')
        # 将模型设置为评估模式
        self.net.eval()

    @torch.no_grad()
    def recognize(self, img, label_map=None):
        # 图像预处理
        img = self.preprocess(img)
        # 进行预测
        y = self.net(img)
        # 计算输出的概率分布
        p_y = torch.nn.functional.softmax(y, dim=1)
        # 获取预测的概率、类别
        p, cls_index = torch.max(p_y, dim=1)
        # 获取预测类别对应的类别名称
        cls_name = label_map[cls_index]
        return cls_name, p.item()

    def preprocess(self, img):
        # 定义图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize
            transforms.ToTensor(),  # 转为tensor
            transforms.Normalize(mean=[0.4737, 0.4948, 0.4336], std=[0.1920, 0.1592, 0.2184])  # 归一化
        ])
        # 图像预处理并转换为形状为(1, C, H, W)的张量
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor


if __name__ == '__main__':
    # 模型路径
    model_path = 'weight.pth'
    # 创建预测类
    recognizer = Predict(model_path)
    # 标签映射表
    label_map = ['番茄叶斑病', '苹果黑星病', '葡萄黑腐病']

    # 测试图片路径
    img_path = 'test/番茄叶斑病/番茄叶斑病 (25).JPG'
    print(f'预测单张图片:{img_path}')
    # 打开测试图片
    img = Image.open(img_path)
    # 进行预测
    st = time.time()
    cls_name, p = recognizer.recognize(img, label_map)
    # 输出预测结果
    print(f'推理时间:{time.time() - st}\t真实标签:{img_path}\t预测标签:{cls_name}\t预测概率:{p}')
