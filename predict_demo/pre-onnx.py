import onnxruntime
import numpy as np
from PIL import Image
import cv2

# 加载模型
model_path = '../model.onnx'
session = onnxruntime.InferenceSession(model_path)


# 预处理图片
def preprocess(img):
    # img = img.resize((224, 224))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    # 0~255 ——》 0~1
    # img = img.astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    # c,h,w
    img = np.transpose(img, (2, 0, 1))
    # n,c,h,w
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

    # # # # # 定义图像预处理
    # from torchvision import transforms
    # from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
    # transform = transforms.Compose([
    #     # transforms.Resize((224, 224), interpolation=cv2.INTER_LINEAR),  # resize
    #     # transforms.Resize((224, 224)),  # resize
    #     CV2_Resize((224, 224), interpolation=cv2.INTER_LINEAR),
    #     transforms.ToTensor(),  # 转为tensor
    #     # transforms.Normalize(mean=[0.4737, 0.4948, 0.4336], std=[0.1920, 0.1592, 0.2184])  # 归一化
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #
    # ])
    # # 图像预处理并转换为形状为(1, C, H, W)的张量
    # img_tensor = transform(img).unsqueeze(0)
    #
    # return img_tensor.numpy()


# 标签映射表
label_map = ['番茄叶斑病', '苹果黑星病', '葡萄黑腐病']

# 测试图片路径
img_path = 'tomato25.JPG'
print(img_path)
# 打开测试图片1
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 打开测试图片2
# img = Image.open(img_path)

# 进行预测
img_tensor = preprocess(img)
outputs = session.run(None, {'data': img_tensor})  # 模型输出
print(outputs)
print(np.exp(outputs) / np.sum(np.exp(outputs)))
output = outputs[0][0]
pred_index = np.argmax(output)  # 最大值的索引
pred_class = label_map[pred_index]
pred_score = np.exp(output[pred_index]) / np.sum(np.exp(output))  # 转概率

# 输出预测结果
print(f'预测标签：{pred_class}')
print(f'预测分数：{pred_score}')

# # 预测多张图片
# folder_path = './predict_demo/test-cv'
# files = os.listdir(folder_path)
# # 得到每个img文件地址
# images_files = [os.path.join(folder_path, f) for f in files]
# for img in images_files:
#     true_label = img.split('\\')[-1].split('.')[0]
#     # print(img)
#     imgs = os.listdir(img)
#     img_path = [os.path.join(img, f) for f in imgs]
#     for img_path in img_path:
#         # print(img_path)
#         # print(img,true_label)
#         image = cv2.imread(img_path)
#         image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         import time
#         st = time.time()
#         # # 进行预测
#         img_tensor = preprocess(image)
#         outputs = session.run(None, {'data': img_tensor})    # 模型输出
#         print(outputs)
#         output = outputs[0][0]
#         pred_index = np.argmax(output)  # 最大值的索引
#         pred_class = label_map[pred_index]
#         pred_score = np.exp(output[pred_index]) / np.sum(np.exp(output))    # 转概率        if true_label == cls:
#         print(f'推理时间:{time.time() - st}\t真实标签:{true_label}\t 预测标签:{pred_score}\t 预测概率:{pred_score}')
