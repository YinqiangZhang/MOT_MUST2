% initialize a MOT tracker
function tracker = initialize_tracker(fr, frame_image, id, bboxes, ind, tracker, opt)

if tracker.state ~= opt.STATE_START
    return;
else
    % build the bboxes structure
    bbox.fr = bboxes.fr(ind);
    bbox.id = id;
    bbox.x = bboxes.x(ind);
    bbox.y = bboxes.y(ind);
    bbox.w = bboxes.w(ind);
    bbox.h = bboxes.h(ind);
    bbox.r = bboxes.r(ind);
    bbox.state = opt.STATE_ACTIVATED;
    % 传递为ECO的初始值
    bbox.eco_x = bbox.x;
    bbox.eco_y = bbox.y;
    bbox.eco_w = bbox.w;
    bbox.eco_h = bbox.h;
    
    % 初始化ECO tracker
    tracker = ECO_initialize(tracker, fr, id, bbox, frame_image);
    % 初始化时候直接将 ECO 的状态变为 activated
    tracker.state = opt.STATE_ACTIVATED; 
    tracker.num_occluded = 0;
    tracker.num_tracked = 0;
    tracker.bboxes = bbox;
end