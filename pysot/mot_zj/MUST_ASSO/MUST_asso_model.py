import os
import re
import numpy as np
import scipy.io as sio
import cv2
from os.path import expanduser
import mot_zj.MUST_ASSO.judge_file as jf
import socket
from mot_zj.MUST_ASSO.model import ft_net
from PIL import Image
import torch
import image
import torchvision.transforms as transforms
import mot_zj.MUST_ASSO.Spatial_Attention.reid.feature_extraction as fe
from mot_zj.MUST_ASSO.Spatial_Attention.reid import models
from mot_zj.MUST_ASSO.Spatial_Attention.reid.utils.serialization import load_checkpoint, save_checkpoint
import mot_zj.MUST_ASSO.function_new2 as f
import time

data_transforms = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class AssociationModel(object):
    def __init__(self, args):
        self.time_steps = args.step_times
        # model parameter setting:
        data_name = 'cuhk_detected'
        self.model = models.create('resnet18', num_features=256, dropout=0.5, num_classes=767)
        checkpoint = load_checkpoint(os.path.join(os.getcwd(), 'weights', 'checkpoint.pth.tar'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.frame_root = os.path.join(os.getcwd(), 'result', 'img')
        self.tracklet_root = os.path.join("pysot","img_traj")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print('load weights done!')

    def __call__(self, bboxes_asso, seq_name, frame, id_num):
        # kk = time.time()
        img_trajs = []
        frame_path = os.path.join(self.frame_root, seq_name, "img1", "{:06d}.jpg".format(frame))
        traj_dir = os.path.join(self.tracklet_root, seq_name, str(id_num))
        
        img_frame = cv2.imread(frame_path) # the whole frame image needed
        subfiles = os.listdir(traj_dir)
        subfiles = sorted(subfiles, key=lambda x: int(re.split('\.', x)[0]))
        num_asso = bboxes_asso.shape[0]

        img_traj_list = []
        for subfile in subfiles:
            if subfile[-3:] == 'jpg':
                img_traj_list.append(subfile)
        num_traj = len(img_traj_list)
        if num_traj < self.time_steps:
            tmp_list = img_traj_list[::-1]
            while len(img_traj_list) < self.time_steps:
                img_traj_list += tmp_list
            img_traj_list = img_traj_list[0:self.time_steps]
        else:
            gap = num_traj // self.time_steps
            mod = num_traj % self.time_steps
            tmp_list = img_traj_list
            img_traj_list = []
            for i in range(mod, num_traj, int(gap)):
                img_traj_list.append(tmp_list[i])

        for i in range(self.time_steps):
            img = cv2.imread(os.path.join(traj_dir, img_traj_list[i]))
            img_traj = Image.fromarray(img)
            img_traj = data_transforms(img_traj)
            img_trajs.append(img_traj)
        for i in range(len(img_trajs)):
            s = img_trajs[i].size()
            img_trajs[i] = img_trajs[i].view(-1, s[0], s[1], s[2])
            if i == 0:
                img_tracking = img_trajs[i]
            else:
                img_tracking = torch.cat((img_tracking,img_trajs[i]),dim=0)
        prediction = np.zeros(num_asso, dtype=np.float32)
        # print("pre time:{}s".format(time.time()-kk))
        # association score
        for ii in range(num_asso):
            x1, y1, w, h = np.ceil(bboxes_asso[ii, :].reshape(bboxes_asso[ii, :].size))
            x2 = x1 + w
            y2 = y1 + h
            x1, y1 = np.maximum(0, (x1, y1))
            x2 = np.minimum(img_frame.shape[1], x2)
            y2 = np.minimum(img_frame.shape[0], y2)

            img_det = img_frame[int(y1):int(y2), int(x1):int(x2), :]
            img_det = Image.fromarray(img_det)
            img_det = data_transforms(img_det)
            img_det_size = img_det.size()
            img_det = img_det.view(-1,img_det_size[0],img_det_size[1],img_det_size[2])
            output = f.detection_tracking_com(self.model, img_det, img_tracking, self.time_steps)
            # save the results
            prediction[ii] = output
        #print(prediction)
        return prediction
        
if __name__ == "__main__":
    asso_model = AssociationModel()