import os, sys
sys.path.append("./YOLOv3")
import numpy as np
from cv2 import cv2
from detector import YOLOv3

# video config
video_root = "./TrackingCode/data/MOT_ZJ/level2_video"
videos = os.listdir(video_root)

# yolov3 config
folder = './YOLOv3'
yolo3 = YOLOv3(folder+"/cfg/yolo_v3.cfg",folder+"/yolov3.weights",folder+"/cfg/coco.names", 
                    is_xywh=True, conf_thresh=0.5, nms_thresh=0.4)

yolo3_plot = YOLOv3(folder+"/cfg/yolo_v3.cfg",folder+"/yolov3.weights",folder+"/cfg/coco.names", 
                is_plot=True)


COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = COLORS_10[id%len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

        # # FOR PLOT!!!
        # cv2.namedWindow("yolo3", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("yolo3", 600,600)
        # cv2.imshow("yolo3",ori_img[:,:,(2,1,0)])
        # cv2.waitKey(0)

def xcycwh2xyxy(xcycwh):
    xyxy = np.zeros(xcycwh.shape)
    for i,box in enumerate(xcycwh):
        xyxy[i,0] = box[0]-box[2]//2
        xyxy[i,2] = box[0]+box[2]//2
        xyxy[i,1] = box[1]-box[3]//2
        xyxy[i,3] = box[1]+box[3]//2
    return xyxy

def xcycwh2ltwh(xcycwh):
    ltwh = np.zeros(xcycwh.shape)
    for i,box in enumerate(xcycwh):
        ltwh[i,0] = box[0]-box[2]//2
        ltwh[i,1] = box[1]-box[3]//2
        ltwh[i,2] = box[2]
        ltwh[i,3] = box[3]
    return ltwh


for video in videos:
    video_path = os.path.join(video_root, video)
    cap = cv2.VideoCapture(video_path)
    frame_index = 1

    data = []
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")

    zj_path = "./TrackingCode/data/MOT_ZJ/train"

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

            bbox_xcycwh, cls_conf, cls_ids = yolo3(frame)
            if bbox_xcycwh is not None:
                # select class person,检测目标为人
                mask = cls_ids==0
                bbox_xcycwh = bbox_xcycwh[mask]
                cls_conf = cls_conf[mask]
                
                # txt save
                if len(bbox_xcycwh) > 0:
                    identities = np.array(list(range(len(bbox_xcycwh))))
                    bbox_xyxy  = xcycwh2xyxy(bbox_xcycwh)
                    bbox_ltwh = xcycwh2ltwh(bbox_xcycwh)

                    # image show
                    # draw_bboxes(ori_img, bbox_xyxy,identities)
                    
                    for i, person in enumerate(bbox_ltwh):
                        current_item = [frame_index, -1] + list(person) + [cls_conf[i],-1,-1,-1]
                        frame_item.append(current_item)

        frame_index += 1


    frame_item = np.array(frame_item)
    np.savetxt(os.path.join(det_savepath,"det.txt"), frame_item, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")
    print("%s  finished"%video)
    cap.release()
                    
                