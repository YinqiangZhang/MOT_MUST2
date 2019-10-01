import argparse
from sys import platform
import numpy as np
import os
import sys

from models import *  # set ONNX_EXPORT in models.py
root = os.getcwd()
yolo_path = os.path.join(root,'detector','yolov3')
sys.path.append(yolo_path)
print(root,yolo_path)
from utils.datasets import *
from utils.utils import *





def detect(save_txt=True, save_img=False, stream_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half = opt.output, opt.source, opt.weights, opt.half
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    streams = 'streams' in source and source.endswith('.txt')
    sep_sign = os.sep
    #print('seperation sign is ',os.sep)
    save_res = []
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    #device = torch_utils.select_device(device='cpu')

    # device = torch_utils.select_device(device='cpu')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()
    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if streams:
        stream_img = False
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    elif webcam:
        stream_img = True
        dataset = LoadWebcam(source, img_size=img_size, half=half)
    else:
        save_img = save_img
        print("if source exits",os.path.exists(source))
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)


        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            if streams:  # batch_size > 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s
                print('path',path)
                im_name,_ = path.split('.')
                print(im_name)
                _,_,video_name,_,im_num = im_name.split(sep_sign)
                print('im_number is {}'.format(im_num))
            print(p)

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        # save_res.append([int(im_num),-1,*xyxy,conf,-1,-1,-1])
                        #with open(save_path + '.txt', 'a') as file:
                            #file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                        with open(os.path.join(out, "det"+video_name+'.txt'),'a') as f:
                            xyxy[2] = xyxy[2] - xyxy[0]
                            xyxy[3] = xyxy[3] - xyxy[1]
                            f.write(('%g ' * 10 + '\n') % (int(im_num),-1,*xyxy,conf,-1,-1,-1))
                            #f.write(('%g ' * 9 + '\n') % ( -1, *xyxy, conf, -1, -1, -1))

                    if save_img or stream_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            #if save_txt:
            #    pass#np.savetxt(os.path.join(out, "det.txt"), save_res, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")
            # Stream results
            if stream_img:
                cv2.imshow(p, im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    if save_txt:
        save_form = np.loadtxt(os.path.join(out, "det"+video_name+'.txt'))
        np.savetxt(os.path.join(out, video_name+'.txt'),save_form,fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=yolo_path+'/'+'cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default=yolo_path+'/'+'citypersons.data', help='citypersons.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolo_weights.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')

    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--save_txt',action='store_true',help='determine whether to save txt for det result')
    parser.add_argument('--save_img',action='store_true',help='determine whether to save txt for det result')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.save_txt,opt.save_img)
