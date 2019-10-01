from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from mot_zj.MUST_utils import find_candidate_detection, calc_overlap_occlusion, motion_predict, find_detections_for_association
from mot_zj.MUST_utils import result_leverage, traj_interpolate
import cv2
import numpy as np
import os

class MUSTTracker(object):
    """
    The Wapper used to implenment SiamsRPN++ tracker and as the memory of other state variables 
    """
    def __init__(self, model):
        ############################################################
        # define some overall variables for multi-objects tracking #
        ############################################################
        # SiamsRPN++ tracker and current states
        self.tracker = SiamRPNTracker(model)
        # self.tracker = SiamRPNLTTracker(model)
        # the state of current tracker
        self.track_state = cfg.STATE.START
        # the current tracking bbox
        self.track_bbox = np.array([])
        # frames when this tracker exists
        self.frames = []
        ############################################################
        # bboxes records when this tracker exists
        self.tracking_bboxes = np.array([])
        # state records when this tracker exists (state of each bbox)
        self.bboxes_state = []
        ############################################################
        # the number of frames that this tracker has tracked
        self.num_tracked = 0
        # the number of frames that this tracker has while tracking
        self.num_occluded = 0
        # bounding boxes overlap flag (judge the overlap state between)
        self.bb_overlaps = np.ones((10,))

    def init(self, img, bbox, id_num, frame, seq_name, asso_model):
        # check if the tracker is correctly created
        if self.track_state is not cfg.STATE.START:
            raise 
        self.track_state = cfg.STATE.ACTIVATED
        self.id_num = id_num
        # save the tracked frame index
        self.frames.append(frame)
        # init the SiamRPN++ tracker (resnet50)
        self.tracker.init(img, bbox)
        if len(self.tracker.model.zf) > 1:
            self.template = [zf.clone().cpu() for zf in self.tracker.model.zf]
        else:
            self.template = self.tracker.model.zf.clone().cpu()
        self.track_bbox = bbox[np.newaxis, :]
        # init bbox ratio
        self.template_ratio = bbox[2]/bbox[3]
        
        # set the tracked frame number (the number is set to 1)
        self.num_tracked = 1
        # set the occluded frame number 
        self.num_occluded = 0
        # save the tracking bboxes
        self.tracking_bboxes = self.track_bbox
        # save the state of tracking bboxes
        self.bboxes_state.append(self.track_state)
        # the name of sequence 
        self.seq_name = seq_name
        # the association model
        self.asso_model = asso_model
        # the target can not be to small
        if bbox[2]*bbox[3] <= 1000 and self.seq_name[-1] in ['2', '3', '4', '5']:
            self.track_state = cfg.STATE.STOP
        
        if bbox[2]*bbox[3] <= 1000 and bbox[1] > 200 and self.seq_name[-1] in ['1']:
            self.track_state = cfg.STATE.STOP

    def track(self, img, bboxes_det, frame):
        # the tracking and activated state (in last frame the object is tracked)
        if self.track_state == cfg.STATE.TRACKED or self.track_state == cfg.STATE.ACTIVATED:
            # condition set (track or occlusion)
            self.num_tracked += 1
            self.num_occluded = 0
            ############################################
            #     *** tracking process start***        #
            ############################################
            # implement the SiamRPN++ tracker (resnet50)
            if len(self.template) > 1:
                self.tracker.model.zf = [zf.clone().cuda() for zf in self.template] 
            else:
                self.tracker.model.zf = self.template.clone().cuda() # replace the tracker with its own template
            output = self.tracker.track(img)
            # append the tracking boxes
            result_bbox = np.array(output['bbox'])
            result_score = output['best_score']
            overlap_score = np.mean(self.bb_overlaps) # used to compare with overlap threshold
            # leverage the results between tracker and detections
            result_bbox = result_leverage(result_bbox, bboxes_det)
            # print("Tracker id {} Original score: {}".format(self.id_num, result_score))
            if result_score >= cfg.PARAMS.TRACKING_SCORE_THRESHOLD:
                if self.track_state == cfg.STATE.TRACKED and overlap_score > cfg.PARAMS.OVERLAP_THRESHOLD:
                    track_label = 1
                elif self.track_state == cfg.STATE.ACTIVATED:
                    track_label = 1
                else:
                    track_label = -1
            else:
                track_label = -1
            
            # the target can not be to small
            if result_bbox[2]*result_bbox[3] <= 1000 and self.seq_name[-1] in ['2', '5']:
                track_label = -1

            # judge the state of tracker: 
            if track_label > 0:
                if self.num_tracked > 0.25*30:
                    self.track_state = cfg.STATE.TRACKED
                else:
                    self.track_state = cfg.STATE.ACTIVATED
            else: # tracked -> lost, activated -> stop
                if self.track_state == cfg.STATE.ACTIVATED:
                    self.track_state = cfg.STATE.STOP
                    print("Target {} end.".format(self.id_num))
                else:
                    self.track_state = cfg.STATE.LOST

            # save the result as attribute
            if self.track_state == cfg.STATE.TRACKED or self.track_state == cfg.STATE.ACTIVATED:
                self.track_bbox = np.expand_dims(result_bbox, axis=0) # use new tracking result
                ratio = result_bbox[2]/result_bbox[3]
                change = ratio/self.template_ratio
                if np.maximum(change, 1/change) > 1.2:
                    self.template_ratio = result_bbox[2]/result_bbox[3]
                    self.tracker.init(img,result_bbox)
                    if len(self.tracker.model.zf) > 1:
                        self.template = [zf.clone().cpu() for zf in self.tracker.model.zf]
                    else:
                        self.template = self.tracker.model.zf.clone().cpu()

                # find whether the track_bbox coincides with detection bbox
                # if the tracking score is confident, we can directly set it as solid result
                if not bboxes_det.size == 0:
                    ov, _ = calc_overlap_occlusion(self.track_bbox, bboxes_det)
                    self.bb_overlaps[:-1] = self.bb_overlaps[1:]
                    if np.max(ov) > 0.5:
                        self.bb_overlaps[-1] = 1
                    else:
                        self.bb_overlaps[-1] = 0
                else:
                    self.bb_overlaps[:-1] = self.bb_overlaps[1:]
                    self.bb_overlaps[-1] = 0
            else:
                pass # track_box will not change 

            # if self.track_state == cfg.STATE.ACTIVATED and self.bb_overlaps[-1] == 0:
            #     self.track_state == cfg.STATE.STOP

            ############################################
            #      *** tracking process end***         #
            ############################################
            # judge the bbox out of the frame size or not (e.g. image.shape = (1920, 1080))
            frame_bbox = np.array([1, 1, img.shape[1], img.shape[0]])
            # calculate the occlusion ratio of the bbox with respect to frame size
            _, occ1 = calc_overlap_occlusion(self.track_bbox, frame_bbox)
            if occ1 < cfg.PARAMS.EXIT_THRESHOLD:
                print("The target exits from the camera view.")
                self.track_state = cfg.STATE.STOP
            # TODO: if not track the object, the state variable update or not ?
            # save relative variables
            self.bboxes_state.append(self.track_state)
            self.frames.append(frame)
            self.tracking_bboxes = np.concatenate((self.tracking_bboxes, self.track_bbox), axis=0)
        
        # the lost state
        elif self.track_state == cfg.STATE.LOST:
            self.num_occluded += 1
            # find a set of detections for association
            bboxes_asso = find_detections_for_association(self, bboxes_det, frame)
            # association prpcess
            if bboxes_asso.size == 0:
                asso_label = -1
            else:
                # association process
                ctrack = motion_predict(frame, self)
                # calculate predicted bbox
                width = self.tracking_bboxes[-1, 2]
                height = self.tracking_bboxes[-1, 3]
                x1 = ctrack[0] - (width - 1) / 2
                y1 = ctrack[1] - (height -1) / 2
                bbox_pred = np.array([x1, y1, width, height])

                o_predict, _ = calc_overlap_occlusion(bbox_pred, bboxes_asso)
                motion_score = np.max(o_predict)
                motion_ind = np.argmax(o_predict)
                # find the last tracked and activated frame index
                indices = []
                for ind, bbox_state in enumerate(self.bboxes_state):
                    if bbox_state == cfg.STATE.TRACKED or bbox_state == cfg.STATE.ACTIVATED:
                        indices.append(ind)
                # the last tracked frame of tracker
                fr_tracker = self.frames[indices[-1]]

                # use deep ReID model to associated tracklet and detections (not implement)
                prediction = self.asso_model(bboxes_asso, self.seq_name, frame, self.id_num)
                ass_score = np.max(prediction)
                asso_ind = np.argmax(prediction)

                if ass_score > cfg.PARAMS.ASSOCIATION_SCORE_THRESHOLD:
                    asso_label = 1
                    bboxes_det_one = bboxes_asso[asso_ind, :]
                    print("Target {} associated by appearance with score {}.".format(self.id_num, ass_score))
                
                elif frame - fr_tracker < 5 and ass_score > 0.4 and motion_score > 0.5:
                    asso_label = 1
                    bboxes_det_one = bboxes_asso[motion_ind, :]
                    print("Target {} associated by motion.".format(self.id_num))
                else:
                    asso_label = -1
                
                # update the parameters when asso_label is equal to 1
                if asso_label == 1:
                    # change the parameters
                    center_x = bboxes_det_one[0] + (bboxes_det_one[2]-1)/2
                    center_y = bboxes_det_one[1] + (bboxes_det_one[3]-1)/2
                    w = bboxes_det_one[2]
                    h = bboxes_det_one[3]
                    self.tracker.center_pos = np.array([center_x, center_y])
                    self.tracker.size = np.array([w, h])
                    if len(self.template) > 1:
                        self.tracker.model.zf = [zf.clone().cuda() for zf in self.template] # replace the tracker with its own template
                    else:
                        self.tracker.model.zf = self.template.clone().cuda()
                    output = self.tracker.track(img)
                    # append the tracking boxes
                    result_bbox = np.array(output['bbox'])
                    result_score = output['best_score']
                    # print("Tracker id {} Modified score: {}".format(self.id_num, result_score))
                    # leverage between tracking and detection
                    ratio = result_bbox[2]/result_bbox[3]
                    change = ratio/self.template_ratio
                    if np.maximum(change, 1/change) > 1.2:
                        self.template_ratio = result_bbox[2]/result_bbox[3]
                        self.tracker.init(img,result_bbox)
                        if len(self.tracker.model.zf) > 1:
                            self.template = [zf.clone().cpu() for zf in self.tracker.model.zf]
                        else:
                            self.template = self.tracker.model.zf.clone().cpu()
                    if not result_score > 0.7:
                        result_bbox = result_leverage(result_bbox, bboxes_det)
            
            if asso_label > 0:

                if self.num_tracked > 0.25 * 30:
                    self.track_state = cfg.STATE.TRACKED
                else:
                    self.track_state = cfg.STATE.ACTIVATED
                
                self.track_bbox = np.expand_dims(result_bbox, axis=0)
                self.bboxes_state.append(self.track_state)
                self.tracking_bboxes = np.concatenate((self.tracking_bboxes, self.track_bbox), axis=0)
                self.frames.append(frame)

                # if self.frames[-1] == frame:
                #     # remove the last item of bbox
                #     self.tracking_bboxes = self.tracking_bboxes[:-1, :]
                #     self.frames = self.frames[:-1]
                #     self.bboxes_state = self.bboxes_state[:-1]
                # try to interpolation the tracklet
                # traj_interpolate(self, result_bbox, frame, 30)

                if not bboxes_det.size == 0:
                    ov, _ = calc_overlap_occlusion(self.track_bbox, bboxes_det)
                    self.bb_overlaps[:-1] = self.bb_overlaps[1:]
                    if np.max(ov) > 0.5:
                        self.bb_overlaps[-1] = 1
                    else:
                        self.bb_overlaps[-1] = 0
                # elif result_score > 0.7:
                #     self.bb_overlaps[:-1] = self.bb_overlaps[1:]
                #     self.bb_overlaps[-1] = 1
                else:
                    self.bb_overlaps[:-1] = self.bb_overlaps[1:]
                    self.bb_overlaps[-1] = 0
            else:
                self.track_state = cfg.STATE.LOST
                self.bboxes_state[-1] = self.track_state
                # else:
                #     self.bboxes_state.append(self.track_state)
                #     self.frames.append(frame)
                #     # TODO: think of mechanism
                #     self.tracking_bboxes = np.concatenate((self.tracking_bboxes, np.expand_dims(self.tracking_bboxes[-1, :], axis=0)), axis=0)

            # finish association
            if self.track_state == cfg.STATE.TRACKED or self.track_state == cfg.STATE.ACTIVATED:
                self.num_occluded = 0
            
            # terminate if lost for a long time
            if self.num_occluded > cfg.PARAMS.TERMINATION_THRESHOLD * 30:
                self.track_state = cfg.STATE.STOP
                print("target {} exits due to long time occlusion".format(self.id_num))

           # judge the bbox out of the frame size or not (e.g. image.shape = (1920, 1080))
            frame_bbox = np.array([1, 1, img.shape[1], img.shape[0]])
            # calculate the occlusion ratio of the bbox with respect to frame size
            _, occ1 = calc_overlap_occlusion(self.track_bbox, frame_bbox)
            if occ1 < cfg.PARAMS.EXIT_THRESHOLD:
                print("target outside image by checking boarders.")
                tracker.track_state = cfg.STATE.STOP
            
            
    def results_return(self):
        results_bbox = self.tracking_bboxes
        results_frame = np.array(self.frames)
        results_ids = self.id_num * np.ones_like(results_frame)
        results_state = np.array(self.bboxes_state)
        results = np.hstack((results_frame[np.newaxis, :].T, results_ids[np.newaxis, :].T, results_bbox))
        num_results = results.shape[0]
        img_path = os.path.join(os.path.join("pysot","img_traj"), self.seq_name, str(self.id_num))
        if os.path.exists(img_path):
            img_num = len(os.listdir(img_path))
        else:
            img_num = 0
        if num_results >= 8 and img_num >=8:
            indices = []
            for ii, res_state in enumerate(results_state):
                if res_state == cfg.STATE.TRACKED or res_state == cfg.STATE.ACTIVATED:
                    indices.append(ii)
            results = results[indices]
        else:
            results = np.array([])
        return results

