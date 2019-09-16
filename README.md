> 项目地址：https://gitee.com/windandwine/Argus
> 转载请注明出处

# 一、项目简介

fine-tune YOLO v3 + FaceNet进行人脸识别，辨别。

## 1. 项目结构



## 2.权重文件

yolo v3是基于论文作者模型，在Wider Face数据上fine-tuning的，可以到[这里](https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz)下载

facenet权重，请到[这里](https://pan.baidu.com/share/init?surl=LLPIitZhXVI_V3ifZ10XNg)下载，密码**12mh**

## 3.yolo v3

YOLO v3的详细预测和训练，可到本人另一个项目[YOLO_v3_tensorflow](https://gitee.com/windandwine/YOLO_v3_tensorflow)了解。

# 三、使用方法

项目需要安装tensorflow、opencv-python、numpy。

## 1.制作自己的人脸数据集

截取需要识别的人物的脸部图片，一人一个文件夹，放在路径**data/base_face/**下。

## 2.使用工具将图片转换成128维向量并存储

运行**preprecessing/pre_tools.py**内的**save_vector_csv()**，将图片使用facenet转换为128d向量，并存储为**data/base_face/vector.csv**中。

## 3.训练svm分类器

基于已经储存的vector.csv文件，运行**preprecessing/pre_tools.py**内的**train_face_svm()**，使用scikit-learn训练svm模型，并储存在**data/weights_svm/svm.pkl**中。

## 4.开始测试

运行根目录下的**test,.py**文件。

另外，可以根据需要更改配置**setting/yolo_args.py**和**setting/yolo_args.py**