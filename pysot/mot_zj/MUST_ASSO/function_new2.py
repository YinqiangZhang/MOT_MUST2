from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable
import torch.nn.functional as F

import torch
import datetime


def extract_cnn_feature(model, inputs, modules=None, return_mask = False):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        tmp = model(inputs)
        outputs = tmp[0]
        outputs = outputs.data.cpu()
        if return_mask:
            mask = tmp[4]
            mask = mask.data.cpu()
            return outputs, mask
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def detection_tracking_com(model, det_img, track_img,time_step=4):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        det_img = Variable(det_img.cuda())
        track_img = Variable(track_img.cuda())
    t1 = datetime.datetime.now()
    ff_det_img = extract_cnn_feature(model, det_img)
    ff_tracking_imgs = extract_cnn_feature(model, track_img)
    t2 = datetime.datetime.now()
    # print('the time in network is {}'.format(t2-t1))
    # print(ff_tracking_imgs.size())
    query = ff_det_img.view(1, -1)
    query = F.normalize(query,p=2,dim=1)
    ff_tracking_imgs = ff_tracking_imgs.view(time_step, -1)
    ff_tracking_imgs = F.normalize(ff_tracking_imgs, p=2, dim=1)
    score = torch.mul(ff_tracking_imgs, query)
    score = score.sum(1)
    score = score.cpu()
    score = score.numpy()
    weights = score/score.sum()
    score = score * weights
    score = score.sum()
    return score


def detection_tracking_com_v1(model, det_imgs, track_img,time_step=4):
    use_gpu = torch.cuda.is_available() and False
    det_num = det_imgs.size()
    # print('the number of detected objects is {}'.format(det_num))

    det_num = det_num[0]
    imgs = torch.cat((det_imgs,track_img))
    if use_gpu:
        imgs = Variable(imgs.cuda())
        #track_img = Variable(track_img.cuda())
    else:
        imgs = Variable(imgs)
    t1 = datetime.datetime.now()
    ff_imgs = extract_cnn_feature(model, imgs)
    t2 = datetime.datetime.now()
    # print('the time in network is {}'.format(t2-t1))
    # print('the size of all feature map is {}'.format(ff_imgs.size()))
    #ff_tracking_imgs = extract_cnn_feature(model, track_img)
    #print(ff_tracking_imgs.size())
    """
    query = ff_imgs[:det_num].view(det_num, -1)
    print('the size of query is {}'.format(query.size()))
    query = F.normalize(query,p=2,dim=1)
    ff_tracking_imgs = ff_imgs[det_num:].view(8, -1)
    print('the size of feature map is {}'.format(ff_tracking_imgs.size()))
    ff_tracking_imgs = F.normalize(ff_tracking_imgs, p=2, dim=1)
    scores = []
    for i in range(det_num):
        score = torch.mul(ff_tracking_imgs, query[i].view(1, -1))
        score = score.sum(1)
        score = score.cpu()
        score = score.numpy()
        weights = score/score.sum()
        score = score * weights
        score = score.sum()
        scores.append(score)
    """
    ff_imgs = ff_imgs.view(det_num+time_step, -1)
    ff_imgs = F.normalize(ff_imgs, p=2, dim=1)
    ff_imgs_t = ff_imgs.t()
    scores = torch.matmul(ff_imgs,ff_imgs_t)
    scores = scores[:,det_num:].numpy()
    import numpy as np
    score_avg = np.expand_dims(np.sum(scores,1),axis=0).transpose()
    print(score_avg)
    score_avg = np.tile(score_avg,time_step)
    print(score_avg)
    weights = scores/score_avg
    scores = scores * weights
    scores = np.sum(scores,axis=1)[:det_num].tolist()
    return scores