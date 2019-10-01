# 之江杯(初赛)---行人多目标跟踪方法运行代码 MUST
---
# 依赖项
- Cuda 8.0
- Cudnn 7.0
- Python 3.7
- Pytorch 1.0.1
- MATLAB R2017b

*环境配置示例*
<pre><code>conda create -n mot anaconda python=3.7
conda activate mot
conda install -c menpo opencv
conda install pytorch torchvision cudatoolkit=8.0 -c pytorch
</code></pre>

# ReID网络训练
(初赛使用，复赛已经更换)
1. 数据集：[Market1501](https://pan.baidu.com/s/1ntIi2Op)(MIT license)，相关说明在文件夹Market1501的readme.txt
2. 数据集准备：
将数据集Market1501下载并保存至"./ReID/Market1501"到根目录中。运行prepare_market.py将数据集的结构修改成训练需要的形式（新的文件格式存在pytorch文件夹中，形成了训练用的train和val的文件夹，和测试用的query，gallery，multi-query的文件夹） 
<pre><code>python prepare_market.py
</code></pre>

3. 训练：修改data_dir的路径，直接进行训练，可以修改参数，本次比赛使用参数在opts.yaml中。每次训练使用训练代码，数据处理文件，loss的变化情况以及当次实验用的参数也储存（备份）在"./ReID/model/ft_ResNet50"

4. 验证：
Test.py负责将测试数据（测试中只用query和gallery的数据进行验证）的特征提取出来并保存，方便后面评价方式rank@1，rank@5，rank@10，mAP的计算。
Evaluate.py负责计算上述评价方式。
（初赛的DMAN + ECO算法，参与视频b3的跟踪，使用的ReID方法与复赛相同，详情参见复赛说明）

   
# 使用方法
1. 下载 [ReID](https://pan.baidu.com/s/16zK8NaC4HRM4o_FcNTzRBQ) 已训练神经网络模型,保存到"./ReID/model/ft_ResNet50/"目录下
2. 下载之江杯测试视频，保存到"./TrackingCode/data/MOT_ZJ/level1_vedio/"目录下
3. 下载[Yolov3](https://pjreddie.com/media/files/yolov3.weights)权重文件，保存到"./YOLOv3/"文件夹
4. 运行detect_video_YOLO.py生成标准格式的测试数据文件，位于 "./TrackingCode/data/MOT_ZJ/train/" 文件夹中
<pre><code>python detect_video_YOLO.py
</code></pre>
5. 进入 "./TrackingCode/ECO/" 文件夹中, 运行脚本 install.m 编译ECO跟踪器
6. 运行服务器脚本
<pre><code>python similarity_calculate.py
</code></pre>
6. 在MATLAB中运行客户端脚本 MUST_demo.m 跟踪结果位于"./TrackingCode/results/"文件夹中
7. 如获取比赛所需格式的txt结果文件，运行脚本文件，结果文件位于 
"./TrackingCode/modified_results/"文件夹中
<pre><code>python resultFormatChange.py
</code></pre>

# References
[1] Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: ECO: Efficient convolution operators for tracking. In: CVPR (2017)

[2] Xiang, Y., Alahi, A., Savarese, S.: Learning to track: Online multi-object tracking by decision making. In: ICCV (2015)

[3] Zhu, J., Yang, H., Liu, N., Kim, M., Zhang, W., Yang, M.: Online Multi-Object Tracking with Dual Matching Attention Networks. In: ECCV (2018)

[4] Redmon J , Farhadi A . YOLOv3: An Incremental Improvement. 2018.

[5] Zheng Z , Zheng L , Yang Y . A Discriminatively Learned CNN Embedding for Person Re-identification[J]. Acm Transactions on Multimedia Computing Communications & Applications, 2016, 14(1)

[6] Liang Zheng*, Shengjin Wang, Liyue Shen*, Lu Tian*, Jiahao Bu, and Qi Tian. Person Re-identification Meets Image Search. Technical Report, 2015.  (*equal contribution)  
