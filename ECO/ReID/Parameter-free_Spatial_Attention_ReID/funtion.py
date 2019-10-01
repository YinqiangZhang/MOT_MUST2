from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable
import torch.nn.functional as F

import torch


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


def detection_tracking_com(model, det_img, track_img):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        det_img = Variable(det_img.cuda())
        track_img = Variable(track_img.cuda())
    ff_det_img = extract_cnn_feature(model, det_img)
    ff_tracking_imgs = extract_cnn_feature(model, track_img)
    print(ff_tracking_imgs.size())
    query = ff_det_img.view(1, -1)
    query = F.normalize(query,p=2,dim=1)
    ff_tracking_imgs = ff_tracking_imgs.view(8, -1)
    ff_tracking_imgs = F.normalize(ff_tracking_imgs, p=2, dim=1)
    score = torch.mul(ff_tracking_imgs, query)
    score = score.sum(1)
    score = score.cpu()
    score = score.numpy()
    weights = score/score.sum()
    score = score * weights
    score = score.sum()
    return score