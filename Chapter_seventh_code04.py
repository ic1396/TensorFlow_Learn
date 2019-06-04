#!/usr/bin/python3
# 第七章  示例代码04  异或操作  7.2.3 练习题
# 7.2.3 练习题 使用生成的模拟数据，完成一个异或模型示例。

import numpy as np
import tensorflow as tf

#############以下为练习（2） one-hot 编码函数开始#####################
from sklearn.preprocessing import OneHotEncoder
def onehot(y,start,end):
    ohe = OneHotEncoder()
    a = np.linspace(start,end-1,end-start)
    b = np.reshape(a,[-1,1]).astype(np.int32)
    ohe.fit(b)
    c = ohe.transform(y).toarray()  
    return c    
##############练习（2） one-hot 编码函数结束#########################
    
learning_rate = 1e-4
n_input = 2   # 输入层节点个数
n_label = 1
n_hidden = 2  # 隐藏层节点个数

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])
# 定义学习参数
weight = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
          'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
          }
biases = {'h1': tf.Variable(tf.zeros([n_hidden])),
          'h2': tf.Variable(tf.zeros([n_label]))
          }
# 定义网络模型，正向结构入口为 x ，输出为 y_pred
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight['h1']), biases['h1']))
# y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2']))
##############练习 （1）开始#######################
# y_pred = tf.nn.relu(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2'])) # 使用 relu
y_pred = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2'])) # 使用 leaky relu
# y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2'])) # 使用 sigmoid
##############练习 （1）结束#########################
# 反向使用均值平方差，使用 AdamOptimizer 优化
loss = tf.reduce_mean((y_pred - y) ** 2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 生成示例数据
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')
# 加载模型
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# 训练 10000 次
for i in range(40000):
    sess.run(train_step, feed_dict={x: X, y: Y})
# 计算预测值，已训练 10000 次
print(sess.run(y_pred, feed_dict={x: X}))
# 查看隐藏层输出
print(sess.run(layer_1, feed_dict={x: X}))