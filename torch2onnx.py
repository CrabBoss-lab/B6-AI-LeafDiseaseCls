import torch
import torchvision
import onnx
from net import Net
import numpy as  np

# 加载PyTorch模型
# model = torchvision.models.resnet18(pretrained=False, num_classes=3)
model = Net()
weight = torch.load('./wandb/run-20230516_115514-ykaakaq3/files/model.pth')
model.load_state_dict(weight)
# model.eval()

# img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.int32)
# img = img.astype(np.float32)
# img = (img / 255. - 0.5) / 0.5  # torch style norm
# img = img.transpose((2, 0, 1))
# img = torch.from_numpy(img).unsqueeze(0).float()
# model.eval()

# 设置模型输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出ONNX模型
torch.onnx.export(model, dummy_input, 'model.onnx', input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=11)
