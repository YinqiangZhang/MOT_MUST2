# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re

def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content)


class Cuhk(object):

    def __init__(self, root):

        self.images_dir = osp.join(root)
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
#        self.camstyle_path = 'bounding_box_train_camstyle'
        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)') #正则化查找xxx_cxxx
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.png')))
        for fpath in fpaths:
            fname = osp.basename(fpath)                         #0002_c1s1_000451_03.jpg ...
            pid, cam = map(int, pattern.search(fname).groups()) #pattern.search(fname).groups()---'0002','1' 转成int类型
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]           #(True)pid（0-750）,(False)不重新排序
            cam -= 1                      #cam（0-5）
            ret.append((fname, pid, cam)) #(True)类似'0002_c1s1_000551_01.jpg', 0, 0
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path)              #[('0002_c1s1_000551_01.jpg', 0, 0),(),(), ...]
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))