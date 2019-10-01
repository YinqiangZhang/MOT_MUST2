#!/usr/bin/env bash

#python3 detect.py --weights weights/best1.pt --source data/samples --output output --img-size 1920 --save_img
python3 detect.py --weights weights/best1.pt --source data/b5 --output output5 --img-size 1024 --save_txt --save_img

python3 detect.py --weights weights/best1.pt --source data/b5 --output output5_ --img-size 1920 --save_txt --save_img


#python3 detect.py --weights weights/best1.pt --source data/samples1 --output output_ --img-size 32#0 --save_txt --save_img