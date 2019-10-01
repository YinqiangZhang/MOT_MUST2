# 之江杯(复赛)---行人多目标跟踪方法运行代码 MUST
---
### 依赖项

### 代码环境配置运行


### 方案说明与算法原理
##### 成绩运行时间说明
- 目标检测：b1\2\3\5(yolov3):共计约2h（1050Max-q, 16G）
- 目标检测：b4(keypoints-Mask-R-CNN):约4h（1080Ti*1, 64G）
- 基于检测结果输出跟踪提交结果：共计约5h（1080Ti*1, 64G）

##### 算法原理
YOLOv3、keypoints-Mask-R-CNN、Siam++可以根据参考文献内容。

###### 算法框架
![之后需要替换](./readme_materials/reid.PNG)


### 数据和预训练权重保存
##### 训练数据
- [cityperson]()
- [cuhk]()

##### 预训练权重
预训练权重保存在weights目录下
- YOLOv3:[darknetXXXX]()
- [keypoints-Mask-R-CNN](https://pan.baidu.com/s/1a8A6xVNuuo6Zr3cc3DbB2Q)
- [pysot]()


### 训练
##### Yolo检测网络训练

##### ReID网络训练

##### 其他
本次比赛中，检测算法使用了[YOLOv3](https://github.com/ultralytics/yolov3)和[keypoints-Mask-R-CNN](https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN)两种方法。

其中，YOLOv3在COCO预训练的基础上进行了重新训练，keypoints-Mask-R-CNN直接使用了作者提供的预训练模型。

为了增强黑夜检测效果的性能，b4采用keypoints-Mask-R-CNN，其他视频检测应用YOLOv3。

### 生成每帧图像和检测保存路径
<pre><code>python detector/to_img.py
</code></pre>

### 目标检测及后处理
<pre><code>FOR b1,b2,b3,b5: python detector/YOLO/video_demo.py
FOR b4: python detector/Mask_R_CNN_Keypoints/video_demo.py
python detector/det_process.py
</code></pre>

### 目标跟踪
##### SiamRPN
使用pysot框架完成单个行人目标的跟踪


### *特别说明*
由于最后提交日时间问题，b3提交结果为level_1阶段应用提交的ECO跟踪算法。复赛的前几次结果也是应用ECO算法提交，即b3与9.12日提交结果一致。


### 参考文献及程序来源
[1] https://github.com/STVIR/pysot

[2] Li B , Wu W , Wang Q , et al. SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks[J]. 2018.

[4] Zhu, J., Yang, H., Liu, N., Kim, M., Zhang, W., Yang, M.: Online Multi-Object Tracking with Dual Matching Attention Networks. In: ECCV (2018)

[5] Redmon J , Farhadi A . YOLOv3: An Incremental Improvement. 2018.

[6] Wang H , Fan Y , Wang Z , et al. Parameter-Free Spatial Attention Network for Person Re-Identification[J]. 2018.

[7] https://github.com/chrispolo/Keypoints-of-humanpose-with-Mask-R-CNN

[8] https://github.com/ultralytics/yolov3