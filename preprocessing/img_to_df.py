# -*- coding:utf-8 -*-

import os
import cv2
import time
import numpy as np
import tensorflow as tf
from net.facenet_model import FaceNet
import setting.facenet_args as facenet_args


def convert_base_face_to_vevtor(facenet):
    """
    获得文件转成128维向量
    :param facenet:
    :return:
    """
    vector_dict = {}
    for dir in os.listdir(facenet_args.base_face_dir):
        real_dir = os.path.join(facenet_args.base_face_dir, dir)  # 每个人的文件夹
        if os.path.isdir(real_dir):
            tag = dir  # 人名
            list = []
            for file in os.listdir(real_dir):  # 文件名
                file_path = os.path.join(real_dir, file)
                img_origin = cv2.imread(file_path)
                img_160 = cv2.resize(img_origin, (160, 160))
                list.append(img_160)
            img_arr = np.stack(tuple(list))  # 拼接图片arr, shape=?*160*160*3
            vector = facenet.img_to_vetor128(img_arr)  # 某人所有图片的128维向量
            vector_dict[tag] = vector
    return vector_dict


def save_vector_npz():
    facenet = FaceNet()
    vector_dict = convert_base_face_to_vevtor(facenet)
    np.savez(facenet_args.base_face_npz, dict=vector_dict)


def watch_npz_file():
    data = np.load(facenet_args.base_face_npz, allow_pickle=True)
    print(data['dict'])


if __name__ == '__main__':
    # save_vector_npz()
    watch_npz_file()
