import os
import re
import cv2
import random 



def extract_and_resize_frames(video_folder):
    # 遍历视频文件夹
    print(os.listdir(video_folder))
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            # 从文件名中提取 s 和 e 的值
            match = re.search(r'_(\d+)_(\d+)\.mp4', filename)
            if match:
                # 构建视频文件的完整路径
                video_path = os.path.join(video_folder, filename)
                # 提取视频文件名（去除扩展名）
                video_name = os.path.splitext(filename)[0]
                # 构建用于保存图片的目标文件夹路径
                output_folder = os.path.join("/home/yjc/Projects/VideoClassify/train_test/train", video_name+"_target")
                

                # 打开视频文件
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"无法打开视频文件: {video_path}")
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                s = int(int(match.group(1)) * cap.get(cv2.CAP_PROP_FPS))
                e = int(int(match.group(2)) * cap.get(cv2.CAP_PROP_FPS))
                edge_s1 = max(s - int((e - s) * 0.5), 0)
                edge_e1 = s
                edge_s2 = e
                edge_e2 = min(e + int((e - s) * 0.5), total_frames)
                remain_frame_extract_interval = int((edge_s1 - 0 + total_frames - edge_e2) / 5) - 1
                no_target_fragment_ori_frame_list = list(range(0, edge_s1)) + list(range(edge_e2, total_frames))

                # 根据正例文件夹是否存在，判断是否已经针对该视频生成过正负例了
                if os.path.exists(output_folder+"_fps"+str(int(fps))):
                    print(filename + " already processed")
                    continue
                else:
                    os.makedirs(output_folder+"_fps"+str(int(fps)))
                
                neg_fold_count = 0  # 控制一个正例视频片段对应多少个负例视频片段
                for _ in range(0, len(no_target_fragment_ori_frame_list), remain_frame_extract_interval):
                    if neg_fold_count >= 5:
                        break

                    extract_fnum = int(301 * fps / 30)
                    if remain_frame_extract_interval <= extract_fnum:
                        extract_fnum = remain_frame_extract_interval
                    sta = random.choice(range(_, (_+1+remain_frame_extract_interval-extract_fnum)))    # 只取300帧
                    neg_output_folder = os.path.join("/home/yjc/Projects/VideoClassify/train_test/train", video_name+"_fps"+str(int(fps))+"_frame"+str(no_target_fragment_ori_frame_list[sta - 1]))
                    if not os.path.exists(neg_output_folder):
                        os.makedirs(neg_output_folder)
                    for frame_num in range(0, extract_fnum, round(fps / 30)):     # 只取300帧,以fps为30来算
                        cap.set(cv2.CAP_PROP_POS_FRAMES, no_target_fragment_ori_frame_list[frame_num + sta - 1])
                        ret, frame = cap.read()
                        if ret:
                            
                            resized_frame = cv2.resize(frame, (368, 224))
                            
                            frame_filename = os.path.join(neg_output_folder, f"{no_target_fragment_ori_frame_list[frame_num + sta - 1]}.png")
                            cv2.imwrite(frame_filename, resized_frame)
                            # print(f"保存帧 {frame_num + sta} 到 {frame_filename}")
                        else:
                            print(f"无法读取帧 {frame_num + sta}")
                    neg_fold_count += 1

                # 遍历指定的帧范围
                for frame_num in range(s, e + fps, round(fps / 30)):
                    # 设置要读取的帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                    ret, frame = cap.read()
                    if ret:
                        
                        resized_frame = cv2.resize(frame, (368, 224))
                            
                        frame_filename = os.path.join(output_folder+"_fps"+str(int(fps)), f"{frame_num}.png")
                        cv2.imwrite(frame_filename, resized_frame)
                        print(f"保存帧 {frame_num} 到 {frame_filename}")
                    else:
                        print(f"无法读取帧 {frame_num}")

                # 释放视频捕获对象
                cap.release()
        # break


# 请将此路径替换为实际的视频文件夹路径
video_folder = '/media/yjc/新加卷/201910/RD清妍/201910_RD清妍_stage1_remain/s2_positive_train1'
extract_and_resize_frames(video_folder)





