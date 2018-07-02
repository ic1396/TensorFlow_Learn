#!/usr/bin/python3
# 第六章  示例代码01  softmax 的应用和交叉熵实验。

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print("scaled = ", sess.run(logits_scaled))
    print("scaled2 = ", sess.run(logits_scaled2))  # 经过第二次的 softmax 后，分布概率会有变化
    print("rell = ", sess.run(result1), "\n")  # 正确的方式
    print("rel2 = ", sess.run(result2), "\n")  # 如果将 softmax 变换完的值放进去，就相当于计算第二次 softmax 的 loss，所以会出错
    print("rel3 = ", sess.run(result3), "\n")

# 标签总概率为 1
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel4 = ", sess.run(result4), "\n")

# sparse 标签
labels = [2, 1]  # 表明 labels 中总共分为3个类：0、1、2。[2，1]等价于 onehot 编码中的 001 与 010。
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel5 = ", sess.run(result5), "\n")

loss = tf.reduce_sum(result1)
with tf.Session() as sess:
    print("loss = ", sess.run(loss))

labels = [[0, 0, 1], [0, 1, 0]]
loss2 = -tf.reduce_sum(labels * tf.log(logits_scaled))
with tf.Session() as sess:
    print("loss2 = ", sess.run(loss2))