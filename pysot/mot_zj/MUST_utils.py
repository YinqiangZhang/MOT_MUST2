from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import sys
sys.path.append('..\\pysot-DMAN')

import numpy as np
from numpy import newaxis
from pysot.core.config import cfg

class BBoxes(object):

    def __init__(self, bboxes=np.array([])):
        super(BBoxes,self).__init__() 
        self.bboxes = bboxes
    
    def _return_frame(self, index):
        return self.bboxes[:, 0] if index == None else self.bboxes[index, 0]
    def _return_id(self, index):
        return self.bboxes[:, 1] if index == None else self.bboxes[index, 1]
    def _return_bbox(self, index):
        return self.bboxes[:, 2:6] if index == None else self.bboxes[index, 2:6]
    def _return_confidence(self, index):
        return self.bboxes[:, 6] if index == None else self.bboxes[index, 6]

    def return_element(self, element, index=None):
        return {
            'fr': self._return_frame(index),
            'id': self._return_id(index),
            'bbox': self._return_bbox(index),
            'score': self._return_confidence(index)
        }[element]
    
    def concatenate(self, bbox):
        self.bboxes = np.concatenate(([self.bboxes], [bbox]), axis=0)

    def __call__(self):
        return self.bboxes

##########################################################################################
##########################################################################################

class StateBBoxes(BBoxes):
    """
    The Container to save the tracking results 
    """
    def __init__(self, bbox, state):
        super(StateBBoxes, self).__init__(bbox)
        self.bboxes_state = np.array(state)

    def _return_state(self, index):
        return self.bboxes_state[index]

    def return_element (self, element, index=None):
        try:
            return super(StateBBoxes, self).return_element(element, index)
        except KeyError:
            return {
                'state': self._return_state(index)
            }[element]

    def concatenate(self, bbox, state):
        super(StateBBoxes,self).concatenate(bbox)
        self.bboxes_state = np.append(self.bboxes_state, state)

##########################################################################################
# draw bboxes function, cited from the Code from DeepSORT
# Github: https://github.com/ZQPei/deep_sort_pytorch
##########################################################################################
# the color table:
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
        x1,y1,w,h = [int(i) for i in box]
        x1 += offset[0]
        x2 = x1 + w - 1 + offset[0]
        y1 += offset[1]
        y2 = y1 + h - 1 + offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = COLORS_10[id%len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

##########################################################################################
##########################################################################################

def find_candidate_detection(trackers, bboxes_det):
    """
    find candidate detections which have not been covered enough by any existing trackers

    return:
    bboxes_det: [x, y, w, h]
    index_det: List -> int 
    """
    if bboxes_det.size == 0:
        index_det = []
        return index_det

    bboxes_tracker = np.array([])

    # select the relative bbox in each tracker
    for ii, tracker in enumerate(trackers):
        if tracker.track_state == cfg.STATE.TRACKED or tracker.track_state == cfg.STATE.ACTIVATED:
            if bboxes_tracker.size == 0:
                bboxes_tracker = tracker.track_bbox
            else:
                bboxes_tracker = np.concatenate((bboxes_tracker, tracker.track_bbox), axis=0)
    
    # compare the results
    num_det = bboxes_det.shape[0]
    if bboxes_tracker.size == 0:
        num_track = 0
    else:
        num_track = bboxes_tracker.shape[0]
    
    if num_track:
        ov, occ = calc_overlap_occlusion(bboxes_det, bboxes_tracker)
        OV = np.max(ov, axis=1)
        OCC = np.sum(occ, axis=1)
        # judge the file 
        indices = [i for i in range(num_det) if OV[i] < 0.5 and OCC[i] < 0.5]
    else:
        indices = range(num_det)
    return indices

##########################################################################################
##########################################################################################

def handle_conflicting_trackers(trackers, bboxes_det):
    """
    solve the conflict of trackers

    return:
    trackers that do not have conflict situations
    """
    # save the active tracker information: bboxes and state
    bboxes_tracker = np.array([])
    bboxes_id = []
    
    for tracker in trackers:
        # select the presenting trackers bboxes
        if tracker.track_state == cfg.STATE.TRACKED or tracker.track_state == cfg.STATE.ACTIVATED:
            # save the corresponding states
            bboxes_id.append(tracker.id_num)
            # save the corresponding bboxes (indices is equal to states)
            if bboxes_tracker.size == 0:
                bboxes_tracker = tracker.track_bbox
            else:
                bboxes_tracker = np.concatenate((bboxes_tracker, tracker.track_bbox), axis=0)            
        # deinit the stop trackers
        if tracker.track_state == cfg.STATE.STOP:
            tracker.tracker = None
            tracker.template = None

    # the number of the detections
    if bboxes_det.size == 0:
        pass
    else:
        if bboxes_tracker.size == 0:
            num_track = 0
        else:
            num_track = bboxes_tracker.shape[0]
        
        # flags that indicate the surpressed tracker
        flag = np.zeros(num_track)
        flag = np.expand_dims(flag, axis=0)
        for ii in range(num_track):

            if flag[:, ii] == 1:
                continue
            _, occ = calc_overlap_occlusion(bboxes_tracker, bboxes_tracker, idx1_list=ii)
            # remove the effects of self overlap
            occ[:, ii] = 0
            # use this, ndarray structure is needed
            occ[flag == 1] = 0

            max_ov = np.max(occ)
            max_ind = np.argmax(occ)

            if max_ov > 0.7: # overlap_sup
                # find the tracked number of relative trackers
                num1 = trackers[bboxes_id[ii]-1].num_tracked
                num2 = trackers[bboxes_id[max_ind]-1].num_tracked
                # judge which bbox have a larger overlap with detections bbox
                ov1, _ = calc_overlap_occlusion(bboxes_tracker, bboxes_det, idx1_list=ii)
                ov2, _ = calc_overlap_occlusion(bboxes_tracker, bboxes_det, idx1_list=max_ind)

                ov1_max = np.max(ov1)
                ov2_max = np.max(ov2)

                if num1 > num2:
                    suppressed_idx = max_ind
                    winner_idx = ii
                elif num1 < num2:
                    suppressed_idx = ii
                    winner_idx = max_ind
                else:
                    if ov1_max > ov2_max:
                        suppressed_idx = max_ind
                        winner_idx = ii
                    else:
                        suppressed_idx = ii
                        winner_idx = max_ind

                if trackers[bboxes_id[suppressed_idx]-1].tracking_bboxes.shape[0] == 1:
                    trackers[bboxes_id[suppressed_idx]-1].track_state = cfg.STATE.STOP
                    trackers[bboxes_id[suppressed_idx]-1].bboxes_state[-1] = cfg.STATE.STOP
                else:
                    trackers[bboxes_id[suppressed_idx]-1].track_state = cfg.STATE.LOST
                    trackers[bboxes_id[suppressed_idx]-1].bboxes_state[-1] = cfg.STATE.LOST
                
                print("target {} suppressed by {}".format(bboxes_id[suppressed_idx], bboxes_id[winner_idx]))
                flag[0, suppressed_idx] = 1

    return trackers

##########################################################################################
##########################################################################################

def sort_trackers(trackers):
    # find the tracked frame length and current tracking state
    frame_length = [] # frame length of each tracker
    tracker_state = [] # state of each tracker
    for tracker in trackers:
        frame_length.append(tracker.num_tracked)
        tracker_state.append(tracker.track_state)
        
    index1 = [ii for ii in range(len(frame_length)) if frame_length[ii] > 10]
    index2 = [ii for ii in range(len(frame_length)) if frame_length[ii] <= 10]
    # find the state group 
    index1 = sorted(index1, key=lambda x: tracker_state[x])
    index2 = sorted(index2, key=lambda x: tracker_state[x])
    # ind1 = np.argsort(state_group1)
    # index1 = [index1[ii] for ii in ind1]
    # ind2 = np.argsort(state_group2)
    # index2 = [index2[ii] for ii in ind2]
    
    return index1, index2

##########################################################################################
##########################################################################################

def calc_overlap_occlusion(bboxes1, bboxes2, idx1_list=None, idx2_list=None):

    """
    caculate the overlap ratios and occluded ratios between
    bboxes1[idx1_list,:] and bboxes2[idx2_list, :]
    """
    # get the selected bounding boxes
    bbs1 = bboxes1.copy() if idx1_list is None else bboxes1[idx1_list, :].copy()
    bbs2 = bboxes2.copy() if idx2_list is None else bboxes2[idx2_list, :].copy()
    
    bbs1 = bbs1[np.newaxis, :] if bbs1.ndim == 1 else bbs1
    bbs2 = bbs2[np.newaxis, :] if bbs2.ndim == 1 else bbs2
    # calculate each area
    area1 = bbs1[:, 2] * bbs1[:, 3]
    area2 = bbs2[:, 2] * bbs2[:, 3]

    # reform the bounding box [x1, y1, w, h] -> [x1, y1, x2, y2]
    bbs1[:, 2] = bbs1[:, 0] + bbs1[:, 2] - 1
    bbs1[:, 3] = bbs1[:, 1] + bbs1[:, 3] - 1

    bbs2[:, 2] = bbs2[:, 0] + bbs2[:, 2] - 1
    bbs2[:, 3] = bbs2[:, 1] + bbs2[:, 3] - 1

    ov = np.empty((bbs1.shape[0], bbs2.shape[0]))
    occ = np.empty((bbs1.shape[0], bbs2.shape[0]))
    # occ2 = np.empty((bbs1.shape[0], bbs2.shape[0]))

    # find the overlap area
    for ii, bb1 in enumerate(bbs1):
        inter_x1 = np.max(np.vstack((bb1[0]*np.ones_like(bbs2[:, 0]), bbs2[:, 0])), axis=0)
        inter_y1 = np.max(np.vstack((bb1[1]*np.ones_like(bbs2[:, 1]), bbs2[:, 1])), axis=0)
        inter_x2 = np.min(np.vstack((bb1[2]*np.ones_like(bbs2[:, 2]), bbs2[:, 2])), axis=0)
        inter_y2 = np.min(np.vstack((bb1[3]*np.ones_like(bbs2[:, 3]), bbs2[:, 3])), axis=0)
        
        # calculate and verify the width and height of intersection region 
        inter_w = inter_x2 - inter_x1 + 1
        inter_h = inter_y2 - inter_y1 + 1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0
        
        # calculate the area of the intersection region
        inter_area = inter_w * inter_h
        # calcualte the area of the union region
        union_area = area1[ii] + area2 - inter_area

        # calculate overlap and occlusion
        ov[ii, :] = inter_area / union_area
        occ[ii, :] = inter_area / area1[ii]
        # occ2[ii, :] = inter_area / area2
    
    return ov, occ 

##########################################################################################
##########################################################################################
def motion_predict(frame_current, tracker):
    # select activated and tracked bboxes
    tracking_bboxes = tracker.tracking_bboxes
    indices = []
    for ind, bbox_state in enumerate(tracker.bboxes_state):
        if bbox_state == cfg.STATE.TRACKED or bbox_state == cfg.STATE.ACTIVATED:
            indices.append(ind)
    # the list can be directly used as index 
    bboxes = tracking_bboxes[indices, :]
    center_x = bboxes[:, 0] + bboxes[:, 2]/2
    center_y = bboxes[:, 1] + bboxes[:, 3]/2
    # find corresponding frames number
    frames_past = []
    for ind, frame in enumerate(tracker.frames):
        if ind in indices:
            frames_past.append(frame)
    # the present frame: (frame_current = frame_current)
    # the total length of frames in the past
    num = len(frames_past)
    # select K lastest bboxes
    K = 10
    if num > K:
        center_x = center_x[num-K:num]
        center_y = center_y[num-K:num]
        frames_past = frames_past[num-K:num]
    
    # compute velocity
    vx = 0
    vy = 0
    num = center_x.size # the number of center pos
    count = 0
    for ii in range(1, num):
        vx = vx + (center_x[ii] - center_x[ii-1]) / (frames_past[ii]-frames_past[ii-1])
        vy = vy + (center_y[ii] - center_y[ii-1]) / (frames_past[ii]-frames_past[ii-1])
        count += 1
    
    if count:
        vx = vx / count
        vy = vy / count
    
    if num == 0:
        bboxes = tracker.tracking_bboxes
        center_x_pred = bboxes[-1, 0] + bboxes[-1, 2]/2
        center_y_pred = bboxes[-1, 1] + bboxes[-1, 3]/2
    else:
        center_x_pred = center_x[-1] + vx * (frame_current - frames_past[-1])
        center_y_pred = center_y[-1] + vy * (frame_current - frames_past[-1])

    return np.hstack((center_x_pred, center_y_pred))

##########################################################################################
##########################################################################################
def find_detections_for_association(tracker, bboxes_det, frame):

    center_pred = motion_predict(frame, tracker)

    if bboxes_det.size == 0:
        return np.array([])
    else:
        center_dets_x = bboxes_det[:, 0] + bboxes_det[:, 2]/2
        center_dets_y = bboxes_det[:, 1] + bboxes_det[:, 3]/2
        center_dets = np.vstack((center_dets_x, center_dets_y)).T
        
        distances = np.linalg.norm(center_dets - center_pred, axis=1) / tracker.tracking_bboxes[-1, 2]
        ratios = tracker.tracking_bboxes[-1, 3] / bboxes_det[:, 3]
        ratios = np.minimum(ratios, 1/ratios)

        # find the quafilied indices
        indices_det = []
        for ii in range(distances.size):
            if distances[ii] < cfg.PARAMS.DISTANCE_THRESHOLD and ratios[ii] > cfg.PARAMS.ASPECT_RATIO_THRESHOLD:
                indices_det.append(ii)
        return bboxes_det[indices_det]
##########################################################################################
##########################################################################################
def result_leverage(result_bbox, bboxes_det):
    """
    to get the final result with the tracking results and the most similar detection result
    ruturn bbox like np.array([x,y,w,h])
    """
    if not bboxes_det.size == 0:
        res_bbox = result_bbox[np.newaxis, :] if result_bbox.ndim == 1 else result_bbox
        bboxes_det = bboxes_det[np.newaxis, :] if bboxes_det.ndim == 1 else bboxes_det
        ov, _ = calc_overlap_occlusion(res_bbox, bboxes_det)
        ov_max = np.max(ov)
        ov_ind_max = np.argmax(ov)
        if ov_max > cfg.PARAMS.DET_OVERLAP_THRESHOLD and np.maximum(res_bbox[0,3], bboxes_det[ov_ind_max, 3]) / np.minimum(res_bbox[0,3], bboxes_det[ov_ind_max, 3]) <= 1.3:
            pos = res_bbox[:, 0:2] + (res_bbox[0, 2:]-1)/2
            target_sz = res_bbox[0, 2:]
            det_pos = bboxes_det[ov_ind_max, 0:2] + (bboxes_det[ov_ind_max, 2:]-1)/2
            det_target_sz = bboxes_det[ov_ind_max, 2:]
            # combined results
            pos = ov_max * det_pos + (1 - ov_max) * pos
            target_sz = ov_max * det_target_sz + (1 - ov_max) * target_sz
            # recovered with bbox
            return np.append(pos-(target_sz-1)/2, target_sz)
        else:
            return result_bbox
    else:
        return result_bbox

##########################################################################################
##########################################################################################
def traj_interpolate(tracker, bbox, frame, fps):
    
    if bbox.size == 0:
        return 
    # select actived bboxes
    traj_base = tracker.tracking_bboxes[:-1,:]
    traj_frames = tracker.frames[:-1]
    traj_bboxes_state = tracker.bboxes_state[:-1]
    indices = []
    for ind, state in enumerate(traj_bboxes_state):
        if state == cfg.STATE.TRACKED or state == cfg.STATE.ACTIVATED:
            indices.append(ind)
    
    # if tracking state is activated and tracked
    if len(indices) is not 0:
        ind = indices[-1] 
        fr1 = tracker.frames[ind]
        fr2 = frame


        if fr2 - fr1 <= fps and fr2 - fr1 > 1:
            traj_base = tracker.tracking_bboxes[:ind+1, :]
            traj_frames = tracker.frames[:ind+1]
            traj_bboxes_state = tracker.bboxes_state[:ind+1]
            # box1, the last active frame
            bbox1 = tracker.tracking_bboxes[ind, :]
            # box2, current frame
            bbox2 = bbox
            for fr in range(fr1+1, fr2):
                inter_bbox =  bbox1 + ((bbox2 - bbox1) / (fr2-fr1)) * (fr-fr1)
                traj_frames.append(fr)
                # set the interpolation bbox to lost
                traj_bboxes_state.append(tracker.track_state)
                traj_base = np.concatenate((traj_base, np.expand_dims(inter_bbox, axis=0)), axis=0)

    tracker.tracking_bboxes = np.concatenate((traj_base, np.expand_dims(bbox, axis=0)), axis=0)
    traj_frames.append(frame)
    tracker.frames = traj_frames
    traj_bboxes_state.append(tracker.track_state)
    tracker.bboxes_state = traj_bboxes_state
##########################################################################################
##########################################################################################
def NMS_selection(results_det, ov_thres):
    # get results bbox and confidence
    results_det = results_det[:, 2:7].copy()
    results_det = results_det[np.argsort(results_det[:, 6])]
    ov, occ1 = calc_overlap_occlusion(results_det, results_det)

    num_det = results_det.shape[0]
    
    


##########################################################################################
##########################################################################################

if __name__ == "__main__":

    a = np.array([1,1,10,10])
    c = np.array([[5,5,10,10], [7,7,9,9]])
    d = np.array([[2,2,11,11],[0,0,9,8]])
    b = np.array([[5,5,10,10],[7,7,9,9],[3,3,2,2]])

    #ov, occ = calc_overlap_occlusion(a,b)
    print(result_leverage(a,d))
    #print(ov)
    #print(occ)
