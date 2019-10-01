python detector/to_img.py

python detector/yolov3/detect.py --source result/img/b1/img1 --output result/img/b1/det --img-size 320 --save_txt
python detector/yolov3/detect.py --source result/img/b2/img1 --output result/img/b2/det --img-size 1920 --save_txt
python detector/yolov3/detect.py --source result/img/b3/img1 --output result/img/b3/det --img-size 608 --save_txt
python detector/yolov3/detect.py --source result/img/b5/img1 --output result/img/b5/det --img-size 1024 --save_txt
python detector/Mask_R_CNN_Keypoints/video_demo.py

python detector/det_process.py