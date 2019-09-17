# -*- coding:utf-8 -*-
import time
import tensorflow as tf
import setting.facenet_args as facenet_args


class FaceNet:
    """
    facenet
    """
    def __init__(self):
        self.meta_path = facenet_args.meta_path
        self.ckpt_path = facenet_args.ckpt_path
        self.sess = tf.Session()
        self.__build_net()

    def __build_net(self):
        """
        加载模型建网络
        :return:
        """
        start_time = time.time()
        # 加载模型
        print('\033[32mBegin importing meta graph..\033[0m')
        saver = tf.train.import_meta_graph(self.meta_path)
        print('\033[32mFinish importing meta graph..\033[0m')
        print('\033[32mBegin resoring model..\033[0m')
        saver.restore(self.sess, self.ckpt_path)
        print('\033[32mFinish resoring model..\033[0m')
        # 获得输入输出tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        model_time = time.time()
        print('\033[32m加载模型时间为{}\033[0m'.format(str(model_time - start_time)))

    def img_to_vetor(self, images):
        """
        将图片转为128维向量
        :param images:
        :return:
        """
        print('\033[32mBegin calculating img vector..\033[0m')
        start_time = time.time()
        # 前向传播计算embeddings
        emb = self.sess.run(
            self.embeddings,
            feed_dict={self.images_placeholder: images, self.phase_train_placeholder: False}
        )
        print('\033[32mFinish calculating img vector, cost time {}..\033[0m'.format(str(time.time() - start_time)))
        return emb
