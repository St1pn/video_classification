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
import shutil
import time, decord
from tqdm import tqdm
import cv2
import multiprocessing as mp
from queue import Empty
from collections import OrderedDict
import time
import threading, queue
from PIL import Image
import concurrent.futures
import numpy as np


# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
epochs = 120        # training epochs
batch_size = 8  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

device = torch.device("cuda")
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
try:
    # 加载非 ResNet 部分的encoder参数
    state_dict_enc = torch.load("/home/yjc/Projects/VideoClassify/video-classification/ResNetCRNN/ResNetCRNN_ckpt/cnn_encoder_epoch1.pth")
    # print(state_dict_enc)
    # for name, param in cnn_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer name: {name}, Shape of weights: {param.data.shape}")
    cnn_encoder.load_state_dict(state_dict_enc)
    print("成功加载非 ResNet 部分的模型参数。")
    # 加载非 ResNet 部分的decoder参数
    state_dict_dec = torch.load("/home/yjc/Projects/VideoClassify/video-classification/ResNetCRNN/ResNetCRNN_ckpt/rnn_decoder_epoch1.pth")
    # print(state_dict_dec)
    # for name, param in rnn_decoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer name: {name}, Shape of weights: {param.data.shape}")
    rnn_decoder.load_state_dict(state_dict_dec)
    print("成功加载非 ResNet 部分的模型参数。")
except Exception as e:
    print(f"加载非 ResNet 部分参数时出现错误：{e}")

# prepare testset
def extract_and_resize_frames(video_folder):
    # 遍历视频文件夹
    print(os.listdir(video_folder))
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            # 从文件名中提取 s 和 e 的值
            match = re.search(r'_(\d+)_(\d+)\.mp4', filename)
            if match:
                video_path = os.path.join(video_folder, filename)
                video_name = os.path.splitext(filename)[0]
                output_folder = os.path.join("/home/yjc/Projects/VideoClassify/train_test/test", video_name)
                
                # 打开视频文件
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"无法打开视频文件: {video_path}")
                    continue

                t1 = time.time()
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                t2 = time.time()

                s = int(int(match.group(1)) * cap.get(cv2.CAP_PROP_FPS))
                e = int(int(match.group(2)) * cap.get(cv2.CAP_PROP_FPS))
                batch_X = []
                batch_y = []
                all_seg_names = []
                all_y = []
                all_y_pred = []
                test_loss = 0
                for f_idx in range(0, total_frames, 64 * round(fps / 30)):
                    
                    cur_seg_folder = output_folder+"_fps"+str(int(fps))+"frame"+str(f_idx)+"to"+str(f_idx+64*round(fps/30)-1)
                    all_seg_names.append(cur_seg_folder)

                    neg_f_count = 0
                    X = []
                    for ff_idx in range(f_idx, f_idx+64 * round(fps / 30), round(fps / 30)):
                        # 交叉超出3/4的算作正例，否则为负例
                        cap.set(cv2.CAP_PROP_POS_FRAMES, ff_idx)
                        ret, frame = cap.read()
                        
                        if ret:
                            if ff_idx not in list(range(s, e)):
                                # 负例视频片段
                                neg_f_count += 1
                            
                            # 待和Image加载png再transfrom的方法比较是否得到的tensor无diff
                            resized_frame = cv2.resize(frame, (368, 224))
                            resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                            resized_frame_rgb = Image.fromarray(resized_frame_rgb)
                            resized_frame_rgb = transform(resized_frame_rgb)
                            #########验证和UCF101_ResNetCRNN_infer_parallel.py顺序执行时，原视频某一位置的帧的经过处理后拿到tensor是否相同
                            if ff_idx == 2080:
                                torch.save(resized_frame_rgb, "2080.pt")
                            #######################################
                            
                            X.append(resized_frame_rgb)
                        else:
                            print(f"无法读取帧 {ff_idx}")
                    
                    t3 = time.time()

                    if len(X) < 64:
                        print("frame num < 64")
                        continue
                    X = torch.stack(X, dim=0)
                    if neg_f_count > 64*3/4:
                        y = torch.LongTensor([0])
                    else:
                        y = torch.LongTensor([1])
                    batch_X.append(X)
                    batch_y.append(y)

                    # start infer
                    if len(batch_X) == 8:
                        with torch.no_grad():
                            X = torch.stack(batch_X, dim=0)
                            y = torch.stack(batch_y, dim=0)
                            print(X.shape, y.shape)

                            X, y = X.to(device), y.to(device).view(-1, )
                            output = rnn_decoder(cnn_encoder(X))

                            loss = F.cross_entropy(output, y, reduction='sum')
                            test_loss += loss.item()                 # sum up batch loss
                            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

                            # collect all y and y_pred in all batches
                            all_y.extend(y)   #  [].extend(torch.tensor([1, 2, 3])) --> [tensor(1), tensor(2), tensor(3)]
                            all_y_pred.extend(y_pred)
                        
                        batch_X = []
                        batch_y = []
                    
                        # break

                    t4 = time.time()
                    print(t2-t1, t3-t2, t4-t3)

                test_loss /= len(all_y)

                # 收集预测错误的样本
                wrong_indices = []
                wrong_c = 0
                for i in range(len(all_y)):
                    if all_y[i].item() != all_y_pred[i].item():
                        wrong_indices.append(i)
                        wrong_c += 1
                print("acc: ", (len(all_y) - wrong_c) / len(all_y))
                wrong_seg_filenames = []
                for _, seg_name in enumerate(all_seg_names):
                    if _ in wrong_indices:
                        wrong_seg_filenames.append(seg_name)

                # compute accuracy
                all_y = torch.stack(all_y, dim=0)
                all_y_pred = torch.stack(all_y_pred, dim=0)
                test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
                # show information
                print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

                # 释放视频捕获对象
                cap.release()





                # 收集预测错误的样本
                frame_list = []
                
                cap = cv2.VideoCapture(video_path)
                # 检查视频是否成功打开
                if not cap.isOpened():
                    print("无法打开视频文件")
                else:
                    for seg_fn in wrong_seg_filenames:
                        start_frame_str = seg_fn.split("frame")[1].split("to")[0]
                        end_frame_str = seg_fn.split("frame")[1].split("to")[1]
                        start_frame = int(start_frame_str)
                        end_frame = int(end_frame_str)
                        
                        for f_idx in range(start_frame, end_frame):
                            # 设置视频的起始帧
                            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                            ret, frame = cap.read()
                            if ret:
                                # 保存帧为 PNG 图片
                                if not os.path.exists(seg_fn):
                                    os.makedirs(seg_fn)
                                frame_filename = os.path.join(seg_fn, f"frame_{f_idx}.png")
                                cv2.imwrite(frame_filename, frame)
                                print(f"保存帧 {f_idx} 为 {frame_filename}")
                            else:
                                print(f"无法读取帧 {f_idx}")

                    # 释放视频捕获对象
                    cap.release()
                

# 请将此路径替换为实际的视频文件夹路径
video_folder = '/media/yjc/新加卷/201910/RD清妍/201910_RD清妍_stage1_remain/s2_positive_test'
extract_and_resize_frames(video_folder)





























    
