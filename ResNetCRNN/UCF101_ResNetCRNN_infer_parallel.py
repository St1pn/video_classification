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
    state_dict_enc = torch.load("/home/yjc/Projects/VideoClassify/video-classification/ResNetCRNN/ResNetCRNN_ckpt/cnn_encoder_epoch10.pth")
    # print(state_dict_enc)
    # for name, param in cnn_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer name: {name}, Shape of weights: {param.data.shape}")
    cnn_encoder.load_state_dict(state_dict_enc)
    print("成功加载非 ResNet 部分的模型参数。")
    # 加载非 ResNet 部分的decoder参数
    state_dict_dec = torch.load("/home/yjc/Projects/VideoClassify/video-classification/ResNetCRNN/ResNetCRNN_ckpt/rnn_decoder_epoch10.pth")
    # print(state_dict_dec)
    # for name, param in rnn_decoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer name: {name}, Shape of weights: {param.data.shape}")
    rnn_decoder.load_state_dict(state_dict_dec)
    print("成功加载非 ResNet 部分的模型参数。")
except Exception as e:
    print(f"加载非 ResNet 部分参数时出现错误：{e}")





import cv2
import multiprocessing as mp
import numpy as np


def read_and_resize_frames(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed_frames = []
    frame_index = start_frame
    while frame_index < end_frame and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (368, 224))
            resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            resized_frame_rgb = Image.fromarray(resized_frame_rgb)
            processed_frames.append(resized_frame_rgb)
        else:
            break
        frame_index += 1
    cap.release()
    return processed_frames


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    
    # 平均分成 processe_num 段
    processe_num = 4
    segment_size = frame_count // processe_num
    segments = [(i * segment_size, (i + 1) * segment_size) if i < (processe_num-1) else (i * segment_size, frame_count) for i in range(processe_num)]

    pool = mp.Pool(processes=processe_num)
    results = []
    for start, end in segments:
        result = pool.apply_async(read_and_resize_frames, args=(video_path, start, end))
        results.append(result)

    pool.close()
    pool.join()

    all_frames = []
    for result in results:
        all_frames.extend(result.get())

    return all_frames


if __name__ == "__main__":
    import time
    t = time.time()
    video_path = '/media/yjc/新加卷/201910/RD清妍/201910_RD清妍_stage1_remain/s2_positive_test/RD清妍_2019-10-03_08-14_60.2min_2_254_263.mp4'
    frames = process_video(video_path)
    print(f"Processed {len(frames)} frames.")
    
    # if not os.path.exists("/home/yjc/Projects/VideoClassify/video-classification/tmppppp"):
    #     os.makedirs("/home/yjc/Projects/VideoClassify/video-classification/tmppppp")
    # for _ in range(2000,2256):
    #     frames[_].save(f"/home/yjc/Projects/VideoClassify/video-classification/tmppppp/{_}_.png", "PNG")
    print(time.time() - t)
    
    # 获取total_frames、fps等
    match = re.search(r'_(\d+)_(\d+)\.mp4', video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    s = int(int(match.group(1)) * cap.get(cv2.CAP_PROP_FPS))
    e = int(int(match.group(2)) * cap.get(cv2.CAP_PROP_FPS))
    print(fps, total_frames, s, e)

    # infer
    file_name = os.path.basename(video_path)
    video_name = os.path.splitext(file_name)[0]
    output_folder = os.path.join("/home/yjc/Projects/VideoClassify/train_test/test", video_name)
    
    t = time.time()
    batch_size = 256
    max_workers = 4
    all_y = []
    all_pred_y = []
    all_seg_framerange = []
    test_loss = 0
    batch_X = []
    batch_y = []
    for i in tqdm(range(0, len(frames), batch_size)):
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            batch = frames[i: i + batch_size]
            batch_results = list(executor.map(transform, batch))
        
        for f_idx in range(0, len(batch_results), 64 * round(fps / 30)):
            #########验证和UCF101_ResNetCRNN_infer.py顺序执行时，原视频某一位置的帧的经过处理后拿到tensor是否相同
            if i == 2048 and f_idx == 64 * round(fps / 30):
                torch.save(batch_results[31], "i_2048_batchresults31.pt")
                torch.save(batch_results[32], "i_2048_batchresults32.pt")
                torch.save(batch_results[33], "i_2048_batchresults33.pt")
                torch.save(batch_results[80], "i_2048_batchresults80.pt")
            #######################################
            
            cur_seg_folder = output_folder+"_fps"+str(int(fps))+"frame"+str(i+f_idx)+"to"+str(i+f_idx+64*round(fps/30)-1)
            all_seg_framerange.append(cur_seg_folder)

            X = []
            neg_f_count = 0
            for ff_idx in range(f_idx, f_idx+64 * round(fps / 30), round(fps / 30)):
                X.append(batch_results[f_idx])
                if (i + ff_idx) not in list(range(s, e)):
                    neg_f_count += 1
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
                    all_pred_y.extend(y_pred)
                batch_X = []
                batch_y = []
                    
    test_loss /= len(all_y)
    # 收集预测错误的样本
    wrong_indices = []
    wrong_c = 0
    for i in range(len(all_y)):
        if all_y[i].item() != all_pred_y[i].item():
            wrong_indices.append(i)
            wrong_c += 1
    print("acc: ", (len(all_y) - wrong_c) / len(all_y))
    wrong_seg_filenames = []
    for _, seg_name in enumerate(all_seg_framerange):
        if _ in wrong_indices:
            wrong_seg_filenames.append(seg_name)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_pred_y = torch.stack(all_pred_y, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_pred_y.cpu().data.squeeze().numpy())
    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
 
    

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
        

    print(time.time() - t)













    
