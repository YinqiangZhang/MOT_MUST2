import os
import numpy as np
import scipy.io as sio
import cv2
from os.path import expanduser
import judge_file as jf
import socket
from model import ft_net
from PIL import Image
import torch
import image
import torchvision.transforms as transforms
import Spatial_Attention.reid.feature_extraction as fe
from Spatial_Attention.reid import models
from Spatial_Attention.reid.utils.serialization import load_checkpoint, save_checkpoint
import funtion as f
# determine whether running on MOT training set or test set
dataset = 'train'  # or 'test'
use_gpu = torch.cuda.is_available()
# communicate with the matlab program using the socket
host = '127.0.0.1'
port = 65431
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_tcp.bind((host, port))
socket_tcp.listen(5)
print('The python socket server is ready. Waiting for the signal from the matlab socket client ...')
connection, adbboxess = socket_tcp.accept()
try:
    # model parameter setting:
    data_name = 'cuhk_detected'
    model = models.create('resnet50', num_features=256, dropout=0.5, num_classes=767)
    checkpoint = load_checkpoint(os.path.join('./Spatial_Attention/logs',data_name, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if use_gpu:
        model = model.cuda()
    print(use_gpu)
    print('load weights done!')

    home = expanduser("~")
    data_transforms = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    while 1:
        flag = connection.recv(1024)
        if not flag:
            break
        elif flag != b'client ok':
            print(flag)
        else:
            time_steps = 8
            img_trajs = []
            print(flag)
            mat = sio.loadmat('mot_py.mat')  # saved by the matlab program
            seq_name = str(mat['seq_name'][0].encode('ascii', 'ignore'))
            seq_name = seq_name[2:-1]
            traj_dir = str(mat['traj_dir'][0].encode('ascii', 'ignore'))
            traj_dir = traj_dir[2:-1]
            frame_id = int(mat['frame_id_double'][0, 0])
            target_id = traj_dir.split('/')[-2]
            x_det = mat['bboxes']['x'][0, 0]
            y_det = mat['bboxes']['y'][0, 0]
            w_det = mat['bboxes']['w'][0, 0]
            h_det = mat['bboxes']['h'][0, 0]
            num_det = x_det.shape[0]
            frame_path = 'data/MOT16/' + str(dataset) + '/' + seq_name + '/img1/' + '{:06d}.jpg'.format(frame_id)
            # data = np.zeros((1, time_steps, 224, 224, 6), dtype=np.float32)
            # img_frame = image.load_img(str(frame_path))
            img_frame = cv2.imread(frame_path)
            # img_frame = np.transpose(img_frame,(2,0,1))
            shape_ = np.shape(img_frame)
            img_w = np.shape(img_frame)[1]
            img_h = np.shape(img_frame)[0]
            subfiles = os.listdir(traj_dir)
            subfiles.sort()
            img_traj_list = []
            for subfile in subfiles:
                if subfile[-3:] == 'jpg':
                    img_traj_list.append(subfile)
            num_traj = len(img_traj_list)
            if num_traj < time_steps:
                tmp_list = img_traj_list[::-1]
                while len(img_traj_list) < time_steps:
                    img_traj_list += tmp_list
                img_traj_list = img_traj_list[0:time_steps]
            else:
                gap = num_traj // time_steps
                mod = num_traj % time_steps
                tmp_list = img_traj_list
                img_traj_list = []
                for i in range(mod, num_traj, int(gap)):
                    img_traj_list.append(tmp_list[i])
            for i in range(time_steps):
                img = cv2.imread(traj_dir + img_traj_list[i])
                #img_traj = img
                #img_traj = cv2.resize(img, (256, 128), interpolation=cv2.INTER_CUBIC)
                # img_traj = np.transpose(img_traj,(2,0,1))
                # img_traj = np.expand_dims(img_traj, axis=0)
                img_traj = Image.fromarray(img)
                #img_det = np.expand_dims(img_det,axis=0)
                img_traj = data_transforms(img_traj)
                img_trajs.append(img_traj)
            for i in range(len(img_trajs)):
                s = img_trajs[i].size()
                img_trajs[i] = img_trajs[i].view(-1, s[0], s[1], s[2])
                if i == 0:
                    img_tracking = img_trajs[i]
                else:
                    img_tracking = torch.cat((img_tracking,img_trajs[i]),dim=0)

                #img_traj = preprocess_input(img_traj)
                # data[0, i, :, :, 3:] = img_traj.copy()
            prediction = np.zeros(num_det, dtype=np.float32)
            for i in range(num_det):
                x1 = int(x_det[i, 0])
                y1 = int(y_det[i, 0])
                w = int(w_det[i, 0])
                h = int(h_det[i, 0])
                x2 = x1 + w
                y2 = y1 + h
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                img_det = img_frame[y1:y2, x1:x2,:]

                a = type(img_det)
                img_det_shape = np.shape(img_det)
                img_det = Image.fromarray(img_det)
                #img_det = np.expand_dims(img_det,axis=0)
                img_det = data_transforms(img_det)
                img_det_size = img_det.size()
                img_det = img_det.view(-1,img_det_size[0],img_det_size[1],img_det_size[2])

                #data[0, :, :, :, 0:3] = preprocess_input(img_det)
                print('trajectory size',img_tracking.size())
                output = f.detection_tracking_com(model,img_det,img_tracking)
                print(output)
                prediction[i] = output
            print(prediction)
            sio.savemat('similarity.mat', {'similarity': prediction})
            connection.sendall(bytes('server ok', encoding='utf-8'))
            print('server ok')
finally:
    connection.close()
    socket_tcp.close()
    print('python server closed.')