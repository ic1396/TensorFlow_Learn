#!/usr/bin/python3
# 第五章  示例代码01  利用 TensorFlow 代码下载 MNIST 数据集。

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print('输入数据：', mnist.train.images)
print('输入数据打 shape：', mnist.train.images.shape)
import pylab
im = mnist.train.images[1]
im = im.reshape(-1, 28)
pylab.imshow(im)
pylab.show(im)
print('输入数据打 shape：', mnist.test.images.shape)
print('输入数据打 shape：', mnist.validation.images.shape)