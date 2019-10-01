from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from model import ft_net, ft_net_dense, PCB, PCB_test

"""
# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
"""
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='./Market1501/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = False  # config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Ten Crop
    # transforms.TenCrop(224),
    # transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.ToTensor()(crop)
    #      for crop in crops]
    # )),
    # transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
    #       for crop in crops]
    # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature_track_traj(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_().cuda()
        else:
            ff = torch.FloatTensor(n, 512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if opt.fp16:
                input_img = input_img.half()
            _, outputs = model(input_img)
            ff = ff + outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        ff = ff.data.cpu().float()
        features = torch.cat((features, ff), 0)
    return features


def feature_extract(model, data):
    features = torch.FloatTensor()
    img = data
    n, c, h, w = img.size()[0],img.size()[1],img.size()[2],img.size()[3]
    if opt.use_dense:
        ff = torch.FloatTensor(n,1024).zero_().cuda()
    else:
        ff = torch.FloatTensor(n,512).zero_().cuda()
    if opt.PCB:
        ff = torch.FloatTensor(n,2048, 6).zero_().cuda()  # we have six parts
    for i in range(2):
        if (i == 1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        if opt.fp16:
            input_img = input_img.half()
        _, outputs = model(input_img)
        ff = ff + outputs
    # norm feature
    if opt.PCB:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    ff = ff.data.cpu().float()
    features = torch.cat((features, ff), 0)
    return features


def score_calculate(q_f,g_f):
    query = q_f.view(-1, 1)
    # print(query.shape)
    score = torch.mm(g_f, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    return index[0]



######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(751)
else:
    model_structure = ft_net(751)

if opt.PCB:
    model_structure = PCB(751)

if opt.fp16:
    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


def detection_tracking_com(model, det_img, track_img):
    ff_tracking_imgs = feature_extract(model, track_img)
    ff_det_img = feature_extract(model, det_img)
    query = ff_det_img.view(-1, 1)
    score = torch.mm(ff_tracking_imgs, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    weights = score/score.sum()
    score = score * weights
    score = score.sum()
    return score

