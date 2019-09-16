# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
from net.facenet_model import FaceNet
import setting.facenet_args as facenet_args

"""
提取base face文件, 保存为128维向量的csv文件
"""

meta_path = '../data/weights_facenet/model-20170512-110547.ckpt-250000.meta'
ckpt_path = '../data/weights_facenet/model-20170512-110547.ckpt-250000'


def convert_base_face_to_vector(facenet):
    """
    获得文件转成128维向量
    :param facenet:
    :return:
    """
    vector_list = []
    for dir in os.listdir('../' + facenet_args.base_face_dir):
        real_dir = os.path.join('../' + facenet_args.base_face_dir, dir)  # 单人的文件夹
        if os.path.isdir(real_dir):
            tag = dir  # 人名
            img_list = []
            for file in os.listdir(real_dir):  # 每张图片
                file_path = os.path.join(real_dir, file)
                img_origin = cv2.imread(file_path)
                img_160 = cv2.resize(img_origin, (160, 160))
                img_list.append(img_160)
            img_arr = np.stack(tuple(img_list))  # 拼接图片arr, shape=?*160*160*3

            vector = facenet.img_to_vetor128(img_arr)  # 某人所有图片的128维向量
            for i in range(vector.shape[0]):
                vec_list = vector[i].tolist()
                vec_list.insert(0, tag)
                vector_list.append(vec_list)
    return pd.DataFrame(vector_list)


def save_vector_csv():
    head = list(range(128))
    head.insert(0, 'name')
    facenet = FaceNet(meta_path, ckpt_path)
    vector_df = convert_base_face_to_vector(facenet)
    vector_df.to_csv('../' + facenet_args.base_face_csv, header=head)


def train_face_svm():
    data = pd.read_csv('../' + facenet_args.base_face_csv, index_col=0)
    names = data.pop('name')
    x = data.values
    y = names.values
    clf = svm.SVC()
    clf.fit(x, y)
    joblib.dump(clf, '../' + facenet_args.svm_path)


def watch_csv_file():
    data = pd.read_csv('../' + facenet_args.base_face_csv, index_col=0)
    names = data.pop('name')
    print(data)


if __name__ == '__main__':
    # save_vector_csv()
    watch_csv_file()
