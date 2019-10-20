import os, sys
import argparse
import numpy as np
from cv2 import cv2


def to_img(video):
    # video config
    video_root = "./video"
    video_path = os.path.join(video_root, video)
    cap = cv2.VideoCapture(video_path)

    # height = cap.get(4)
    # width = cap.get(3)
    frame_index = 1

    data = []
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")


    zj_path = "./result/img"
    os.makedirs(zj_path, exist_ok=True)

    if video[:2] not in os.listdir(zj_path):
        os.mkdir(os.path.join(zj_path, video[:2]))

    vdo_savepath = os.path.join(zj_path, video[:2])
    for i in ['img1','det']:
        if i not in os.listdir(vdo_savepath):
            os.mkdir(os.path.join(vdo_savepath, i))

    img_savepath = os.path.join(vdo_savepath, 'img1')
    det_savepath = os.path.join(vdo_savepath, 'det')

    frame_item = []
    while (success):
        # frame.shape = (1080,1920,3)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ori_img = frame
                  
            path = os.path.join(img_savepath, str(frame_index).zfill(6))
            path = path+".jpg"
            # image save
            cv2.imwrite(path, ori_img[:,:,(2,1,0)])

        print(video,": frame",frame_index)
        frame_index += 1


    print("%s  finished"%video)
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='file name',default="c1.mp4")

    opt = parser.parse_args()
    
    
    to_img(opt.file)