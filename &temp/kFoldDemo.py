import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from sklearn.model_selection import KFold

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建ImageFolder数据集对象
train_dataset = ImageFolder(root='train', transform=ToTensor())
val_dataset = ImageFolder(root='val', transform=ToTensor())

# 创建KFold对象，指定k值为5
kf = KFold(n_splits=5, shuffle=True)

# 定义一些超参数
lr = 0.001
batch_size = 64
epochs = 5

# 定义变量来跟踪最佳模型参数和验证集准确率
best_model_params = None
best_accuracy = 0.0

# 对训练集进行k折交叉验证
for fold, (train_indices, test_indices) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold + 1}')

    # 创建训练集和测试集的数据加载器
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)

    # 创建神经网络模型、损失函数和优化器
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 在测试集上进行模型评估
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.4f}')

        # 如果当前模型的验证集准确率更高，则保存当前模型参数
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_params = model.state_dict()

    # 在每个拆分结束后，保存最佳模型参数
    torch.save(best_model_params, f'best_model_fold_{fold}.pt')