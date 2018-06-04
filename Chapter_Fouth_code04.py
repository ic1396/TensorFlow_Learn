#!/usr/bin/python3
# 第四章  示例代码04  关于模型的保存和载入 线性回归模型的载入

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[], "loss":[]}      #存放批次值和损失值
def moving_average(a, w = 10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]

#生成模拟数据
train_X = np.linspace(-1,1,100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
#显示模拟数据
plt.plot(train_X, train_Y, 'ro', label = 'Original Data')
plt.legend()
plt.show()

#重置图
tf.reset_default_graph()

#创建模型
#占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
#模型参数
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.zeros([1]), name = "bias")

#前向结构
z = tf.multiply(X, W) + b
#反向优化
cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  #梯度下降 Gradient descent

#初始化所有变量
init = tf.global_variables_initializer()
#定义训练参数
training_epochs = 20
display_step = 2

saver = tf.train.Saver()
savedir = "log/"
#启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, savedir + "linermodel.cpkt")
    print("x = 0.2, z = ", sess.run(z, feed_dict = {X: 0.2}))