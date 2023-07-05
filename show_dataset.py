from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomResizedCrop, ColorJitter, RandomGrayscale, RandomCrop
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision


# 加载指定目录下的图像，返回根据切分比例形成的数据加载器
def Load_data(trainDir, valDir, shape=(224, 224), batch_size=128):
    transform_train = Compose([
        Resize(shape),
        RandomCrop(224, padding=20),
        RandomHorizontalFlip(),  # 0.5的进行水平翻转
        RandomVerticalFlip(),  # 0.5的进行垂直翻转
        ToTensor(),  # PIL转tensor
        Normalize(mean=[0.4740, 0.4948, 0.4338], std=[0.1920, 0.1591, 0.2184])
        # 归一化   # 输入必须是Tensor
    ])
    transform_test = Compose([
        # Resize(shape),
        ToTensor(),
        # Normalize(mean=[0.4740, 0.4948, 0.4338], std=[0.1920, 0.1591, 0.2184])
    ])

    # 加载数据集
    train_set = ImageFolder(trainDir, transform=transform_train)
    val_set = ImageFolder(valDir, transform=transform_test)

    # 封装批处理的迭代器（加载器）
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_set.class_to_idx


# 测试
if __name__ == "__main__":
    train_loader, val_loader, class_to_idx = Load_data('./dataset/train', './dataset/val', batch_size=64)


    # 按batch_size可视化
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        img = img.numpy().transpose(1, 2, 0)
        plt.imshow(img)
        # plt.savefig('random_train_image')
        plt.show()


    # 获取随机数据
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # print(images, labels)

    # 展示图像
    imshow(torchvision.utils.make_grid(images))
