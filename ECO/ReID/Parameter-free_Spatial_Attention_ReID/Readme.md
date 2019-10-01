## 使用说明：
使用这个方法，训练直接在terminal里运行main.py加上想要调节的参数，如：
python main.py -d cuhk -b 24 -j 4 --epochs 100 --log ./logs/cuhk_detected/ --step-size 40 --data-dir ../cuhk03-np/detected/
可以用Market1501和cuhk训练，用market1501训练的时候，注意修改num_classes为751,cuhk为767，除了在main中修改，还需要在ReSAnet中修改。

也可以用我训练完的模型，结果存在如下网盘中，其中文件夹一一对应：
链接：https://pan.baidu.com/s/16unW0dypsI6I6dH6_j7H5w 
提取码：b85k 
主要调节的是batch_size 和 lerning rate，我用的bs=24,lr=0。03,可以适当调节

