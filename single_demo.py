import os
import cv2
import sys
import nms.nms as nms
import glob
import time
import torch
import shutil
import numpy as np
import yacs
import argparse


root = os.getcwd()
track_dir = os.path.join(root, 'pysot')
params_dir = os.path.join(root, 'weights')
sys.path.append(root)
sys.path.append(track_dir)

def main(args):
    seq_name = args.seq_name
    # the packages of trackers
    from pysot.core.config import cfg # use the modified config file to reset the tracking system
    from pysot.models.model_builder import ModelBuilder
    # modified single tracker with warpper
    from mot_zj.MUST_sot_builder import build_tracker
    from mot_zj.MUST_utils import draw_bboxes, find_candidate_detection, handle_conflicting_trackers, sort_trackers
    from mot_zj.MUST_ASSO.MUST_asso_model import AssociationModel
    from mot_zj.MUST_utils import traj_interpolate


    dataset_dir = os.path.join(root, 'result')
    seq_type = 'img'
    # set the path of config parameters and
    config_path = os.path.join(track_dir, "mot_zj","MUST_config_file","alex_config.yaml")
    model_params = os.path.join(params_dir, "alex_model.pth")
    # enable the visualisation or not 
    is_visualisation = False
    # print the information of the tracking process or not 
    is_print = True

    results_dir = os.path.join(dataset_dir,'track')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    img_traj_dir = os.path.join(track_dir, "img_traj")
    if os.path.exists(os.path.join(img_traj_dir, seq_name)):
        shutil.rmtree(os.path.join(img_traj_dir, seq_name))

    seq_dir = os.path.join(dataset_dir, seq_type)
    seq_names = os.listdir(seq_dir)
    seq_num = len(seq_names)

    # record the processing time
    start_point = time.time()

    # load config
    # load the config information from other variables
    cfg.merge_from_file(config_path)

    # set the flag that CUDA is available 
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create the tracker model (Resnet50)
    track_model = ModelBuilder()
    # load tracker model
    track_model.load_state_dict(torch.load(model_params, map_location=lambda storage, loc: storage.cpu()))
    track_model.eval().to(device)
    # create assoiation model
    asso_model = AssociationModel(args)

    seq_det_path = os.path.join(seq_dir, seq_name, 'det')
    seq_img_path = os.path.join(seq_dir, seq_name, 'img1')

    # print path and dataset information
    if is_print:
        print('preparing for the sequence: {}'.format(seq_name))
        print('-----------------------------------------------')
        print("detection result path: {}".format(seq_det_path))
        print("image files path: {}".format(seq_img_path))
        print('-----------------------------------------------')

    # read the detection results
    det_results = np.loadtxt(os.path.join(seq_det_path, 'det.txt'), dtype=float, delimiter=',')

    # read images from each sequence
    images = sorted(glob.glob(os.path.join(seq_img_path, '*.jpg')))
    img_num = len(images)

    # the contrainer of trackers
    trackers = []

    # visualisation settings
    if is_visualisation:
        cv2.namedWindow(seq_name, cv2.WINDOW_NORMAL)

    # init(reset) the identifer
    id_num = 0

    # tracking process in each frame
    for nn, im_path in enumerate(images):
        each_start = time.time()
        frame = nn + 1
        img = cv2.imread(im_path)
        print('Frame {} is loaded'.format(frame))

        # load the detection results of this frame
        pre_frame_det_results = det_results[det_results[:,0] == frame]

        # non-maximal surpressing [frame, id, x, y, w, h, score]
        indices = nms.boxes(pre_frame_det_results[:,2:6], pre_frame_det_results[:,6])
        frame_det_results = pre_frame_det_results[indices,:]
        
        # extract the bbox [fr, id, (x, y, w, h), score]
        bboxes = frame_det_results[:, 2:6]

        ############################################
        # ***multiple tracking and associating***  #
        ############################################

        # 1. sort trackers
        index1, index2 = sort_trackers(trackers)

        # 2. save the processed index of trackers
        index_processed = []
        track_time = 0;
        asso_time = 0;
        for k in range(2):
            # process trackers in the first or the second class
            if k == 0:
                index_track = index1
            else:
                index_track = index2
            track_start = time.time()
            for ind in index_track:
                if trackers[ind].track_state == cfg.STATE.TRACKED or trackers[ind].track_state == cfg.STATE.ACTIVATED:
                    indices = find_candidate_detection([trackers[i] for i in index_processed], bboxes)
                    to_track_bboxes = bboxes[indices, :] if not bboxes.size == 0 else np.array([]) 
                    # MOT_track(tracking process)
                    trackers[ind].track(img, to_track_bboxes, frame)
                    # if the tracker keep its previous tracking state (tracked or activated)
                    if trackers[ind].track_state == cfg.STATE.TRACKED or trackers[ind].track_state == cfg.STATE.ACTIVATED:
                        index_processed.append(ind)
            track_time += time.time() - track_start
            asso_start = time.time()
            for ind in index_track:
                if trackers[ind].track_state == cfg.STATE.LOST:
                    indices = find_candidate_detection([trackers[i] for i in index_processed], bboxes)
                    to_associate_bboxes = bboxes[indices, :] if not bboxes.size == 0 else np.array([])
                    # MOT_track(association process)
                    trackers[ind].track(img, to_track_bboxes, frame)
                    # add process flag
                    index_processed.append(ind)
            asso_time += time.time() - asso_start
        ############################################
        #        ***init new trackers ***          #
        ############################################

        # find the candidate bboxes to init new trackers
        indices = find_candidate_detection(trackers, bboxes)

        # process the tracker: init (1st frame) and track mathod (the other frames)
        for index in indices:
            id_num += 1
            new_tracker = build_tracker(track_model)
            new_tracker.init(img, bboxes[index, :], id_num, frame, seq_name, asso_model)
            trackers.append(new_tracker)

        # find conflict of trackers (I need to know what conflict)
        trackers = handle_conflicting_trackers(trackers, bboxes)

        # interpolate the tracklet results
        for tracker in trackers:
            if tracker.track_state == cfg.STATE.TRACKED or tracker.track_state == cfg.STATE.ACTIVATED:
                bbox = tracker.tracking_bboxes[-1, :]
                traj_interpolate(tracker, bbox, tracker.frames[-1], 30)

        ############################################
        #    ***collect tracking results***        #
        ############################################
        
        # collect the tracking results (all the results, without selected)
        if frame % 500 == 0:
            results_bboxes = np.array([])
            for tracker in trackers:
                if results_bboxes.size == 0:
                    results_bboxes = tracker.results_return()
                else:
                    res = tracker.results_return()
                    if not res.size == 0:
                        results_bboxes = np.concatenate((results_bboxes, tracker.results_return()), axis=0)
            # test code segment
            filename = '{}.txt'.format(seq_name)
            results_bboxes = results_bboxes[np.argsort(results_bboxes[:, 0])]
            print(results_bboxes.shape[0])
            # detections filter
            indices = []
            if seq_name == 'b1':
                for ind, result in enumerate(results_bboxes):
                    if result[3] > 540:
                        if result[4]*result[5] < 10000:
                            indices.append(ind)
                results_bboxes = np.delete(results_bboxes, indices, axis = 0)
            np.savetxt(os.path.join(results_dir,filename), results_bboxes, fmt='%d,%d,%.1f,%.1f,%.1f,%.1f')
        ############################################
        #        ***crop tracklet image***         #
        ############################################

        for tracker in trackers:
            if tracker.track_state == cfg.STATE.START or tracker.track_state == cfg.STATE.TRACKED or tracker.track_state == cfg.STATE.ACTIVATED:
                bbox = tracker.tracking_bboxes[-1, :]
                x1 = int(np.floor(np.maximum(1, bbox[0])))
                y1 = int(np.ceil(np.maximum(1, bbox[1])))
                x2 = int(np.ceil(np.minimum(img.shape[1], bbox[0]+bbox[2])))
                y2 = int(np.ceil(np.minimum(img.shape[0], bbox[1]+bbox[3])))
                img_traj = img[y1:y2, x1:x2, :]
                traj_path = os.path.join(img_traj_dir, seq_name, str(tracker.id_num))
                if not os.path.exists(traj_path):
                    os.makedirs(traj_path)
                tracklet_img_path = os.path.join(traj_path, str(tracker.frames[-1]))
                cv2.imwrite("{}.jpg".format(tracklet_img_path), img_traj)
        each_time = time.time() - each_start
        print("period: {}s, track: {}s({:.2f}), asso: {}s({:.2f})".format(each_time, track_time,(track_time/each_time)*100, asso_time, (asso_time/each_time)*100))
        if is_visualisation:
            ##########################################
            # infomation print and visualisation     #
            ##########################################
            # print("THe numger of new trackers: {}".format(len(indices)))
            active_trackers = [trackers[i].id_num for i in range(len(trackers)) if trackers[i].track_state == cfg.STATE.ACTIVATED or trackers[i].track_state == cfg.STATE.TRACKED or trackers[i].track_state == cfg.STATE.LOST]
            print("The number of active trackers: {}".format(len(active_trackers)))
            print(active_trackers)
            anno_img = draw_bboxes(img, bboxes)
            cv2.imshow(seq_name, anno_img)
            cv2.waitKey(1)
        print("The running time is: {} s".format(time.time()-start_point))

    print("The total processing time is: {} s".format(time.time()-start_point))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default='b1')
    parser.add_argument("--step_times", type=int, default=8)
    args = parser.parse_args()
    main(args)
