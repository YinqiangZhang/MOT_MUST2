import os
import numpy as np
import coco
import model as modellib
import visualize
from model import log
import cv2
import time

ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.join(ROOT_DIR,"detector","Mask_R_CNN_Keypoints")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR,"weights","mask_rcnn_coco_humanpose.h5")
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
weights = "./weights"
model_path = os.path.join(weights,"mask_rcnn_coco_humanpose.h5")
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'person']
def cv2_display_keypoint(image,boxes,keypoints,masks,class_ids,scores,class_names,skeleton = inference_config.LIMBS):
    # Number of persons
    N = boxes.shape[0]
    if not N:
        print("\n*** No persons to display *** \n")
    else:
        assert N == keypoints.shape[0] and N == class_ids.shape[0] and N==scores.shape[0],\
            "shape must match: boxes,keypoints,class_ids, scores"
    colors = visualize.random_colors(N)
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        for Joint in keypoints[i]:
            if (Joint[2] != 0):
                cv2.circle(image,(Joint[0], Joint[1]), 2, color, -1)

        # #draw skeleton connection
        # limb_colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
        #                [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255], [170, 170, 0], [170, 0, 170]]
        # if (len(skeleton)):
        #     skeleton = np.reshape(skeleton, (-1, 2))
        #     neck = np.array((keypoints[i, 5, :] + keypoints[i, 6, :]) / 2).astype(int)
        #     if (keypoints[i, 5, 2] == 0 or keypoints[i, 6, 2] == 0):
        #         neck = [0, 0, 0]
        #     limb_index = -1
        #     for limb in skeleton:
        #         limb_index += 1
        #         start_index, end_index = limb  # connection joint index from 0 to 16
        #         if (start_index == -1):
        #             Joint_start = neck
        #         else:
        #             Joint_start = keypoints[i][start_index]
        #         if (end_index == -1):
        #             Joint_end = neck
        #         else:
        #             Joint_end = keypoints[i][end_index]
        #         # both are Annotated
        #         # Joint:(x,y,v)
        #         if ((Joint_start[2] != 0) & (Joint_end[2] != 0)):
        #             # print(color)
        #             cv2.line(image, tuple(Joint_start[:2]), tuple(Joint_end[:2]), limb_colors[limb_index],3)
        mask = masks[:, :, i]
        image = visualize.apply_mask(image, mask, color)
        caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
        cv2.putText(image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color)
    return image

videoname = 'b4'
cap = cv2.VideoCapture('video/%s.mp4'%videoname)
size = (
	int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
	int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
output = cv2.VideoWriter('detector/Mask_R_CNN_Keypoints/output/%s.mp4'%videoname,codec,25.0,size)

i = 0
frame_rate_divider = 1
video_det = np.zeros([1,10])

while(cap.isOpened()):
    stime = time.time()
    ret, frame = cap.read()
    if ret:
        if i % frame_rate_divider == 0:
            results = model.detect_keypoint([frame], verbose=0)
            r = results[0]
         # for one image
            # log("rois", r['rois'])
            # log("keypoints", r['keypoints'])
            # log("class_ids", r['class_ids'])
            # log("keypoints", r['keypoints'])
            # log("masks", r['masks'])
            # log("scores", r['scores'])
            result_frame = cv2_display_keypoint(frame,r['rois'],r['keypoints'],r['masks'],r['class_ids'],r['scores'],class_names)
            
            det_f = np.full((r['rois'].shape[0],1),i+1)
            det_id = np.full((r['rois'].shape[0],1),-1)
            # (y1,x1,y2,x2)
            w = r['rois'][:,3]-r['rois'][:,1]
            h = r['rois'][:,2]-r['rois'][:,0]
            frame_xywh = np.concatenate([r['rois'][:,1][:,None],r['rois'][:,0][:,None],w[:,None],h[:,None]],axis=1)
            det_conf = r['scores'][:,None]
            det_3d = np.full((r['rois'].shape[0],3),-1)
            single_frame = np.concatenate([det_f,det_id,frame_xywh,det_conf,det_3d],axis=1)
            
            # 合并
            video_det = np.concatenate([video_det,single_frame],axis=0)

            output.write(result_frame)
            cv2.imshow('frame', result_frame)
            i += 1
        else:
            i += 1
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_det = video_det[1:]
np.savetxt('result/img/%s/det/det.txt'%videoname, video_det, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")
cap.release()
output.release()
cv2.destroyAllWindows()