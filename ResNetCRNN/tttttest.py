import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os
import re
import cv2
import random 

# video_path = "/media/yjc/新加卷/201910/RD清妍/201910_RD清妍_stage1_remain/RD清妍_2019-10-02_10-41_60.2min_2.mp4"
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"无法打开视频文件: {video_path}")

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# s =  1* fps
# e =  2* fps

# for f_idx in range(0, total_frames, 64):
#     # 交叉超出3/4的算作正例，否则为负例
#     cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
#     ret, frame = cap.read()
#     if ret:
#         # 调整帧图片的大小为 (640, 512)
#         resized_frame = cv2.resize(frame, (368, 224))
#         # 构建要保存的图片文件名
#         frame_filename = "xxxxxxxxxxxxxxxxxx.jpg"
#         # 保存调整大小后的帧图片
#         cv2.imwrite(frame_filename, resized_frame)
#         resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
#         print(type(resized_frame_rgb))
#         print(resized_frame_rgb)
#         break
#     else:
#         print(f"无法读取帧 {1}")

# image = Image.open("xxxxxxxxxxxxxxxxxx.jpg")
# image_array = np.array(image)
# transform = transforms.Compose([transforms.ToTensor()])
# image = transform(image)
# print(type(image))
# print(image)












# import cv2
# from PIL import Image
# import numpy as np

# video_path = "/media/yjc/新加卷/201910/RD清妍/201910_RD清妍_stage1_remain/RD清妍_2019-10-02_10-41_60.2min_2.mp4"
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"无法打开视频文件: {video_path}")

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# s = 1 * fps
# e = 2 * fps

# for f_idx in range(0, total_frames, 64):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
#     ret, frame = cap.read()
#     if ret:
#         resized_frame = cv2.resize(frame, (368, 224))
#         frame_filename = "output_frame.png"
#         cv2.imwrite(frame_filename, resized_frame)
#         resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) # 将 BGR 转换为 RGB
#         resized_frame_rgb = Image.fromarray(resized_frame_rgb)
#         transform = transforms.Compose([transforms.Resize([224, 224]),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         resized_frame_rgb = transform(resized_frame_rgb)

#         image = Image.open(frame_filename)
#         # image_array = np.array(image)
#         transform = transforms.Compose([transforms.Resize([224, 224]),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         image = transform(image)


#         # 比较转换后的 resized_frame_rgb 和 image_array
#         difference = torch.abs(resized_frame_rgb - image)
#         print("差异总和:", torch.sum(difference).item())
#         # difference = np.abs(resized_frame_rgb - image)
#         # print("差异总和:", np.sum(difference))
#         break
#     else:
#         print(f"无法读取帧 {f_idx}")

# cap.release()



# import random

# my_list = [1, 2, 3, 4, 5]
# result = []
# for _ in range(1000):
#     result.append(random.choice(my_list))

# # 统计每个元素出现的次数
# counts = {i: result.count(i) for i in my_list}
# print(counts)




# 验证和UCF101_ResNetCRNN_infer.py顺序执行时，原视频某一位置的帧的经过处理后拿到tensor是否相同
import torch
import torch.nn.functional as F

# 从 .pt 文件中加载 Tensor
tensor1 = torch.load('/home/yjc/Projects/VideoClassify/video-classification/2080.pt')
tensor2 = torch.load('/home/yjc/Projects/VideoClassify/video-classification/i_2048_batchresults31.pt')

# 确保两个 Tensor 具有相同的形状
if tensor1.shape != tensor2.shape:
    print("两个 Tensor 的形状不同，无法直接比较。")
else:
    # 找出元素不同的位置
    diff_mask = tensor1 != tensor2

    # 计算不同元素的数量
    diff_count = diff_mask.sum().item()

    # 计算总元素数量
    total_count = tensor1.numel()

    # 计算差异比例
    diff_ratio = diff_count / total_count

    if diff_count == 0:
        print("两个 Tensor 完全相同。")
    else:
        print(f"两个 Tensor 有差异，差异部分占元素总数的比例为: {diff_ratio * 100:.2f}%。")





    