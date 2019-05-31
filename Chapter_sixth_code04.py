#!/usr/bin/python3
# 第六章  示例代码04  以代码 6-2 为基础，用 Maxout 网络实现 MNIST 分类。
# 以代码 Chapter_sixth_code02.py 为基础，其中
# 使用sparse_softmax_cross_entropy_with_logits函数来运算交叉熵。

import tensorflow as tf  # 导入 tensorflow 库
from tensorflow.examples.tutorials.mnist import input_data
import pylab 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # MNIST 数据集的纬度是 28 * 28 = 784
y = tf.placeholder(tf.float32, [None, 10])  # 数字 0 ~ 9，共 10 个类别
# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
z= tf.matmul(x, W) + b

maxout = tf.reduce_max(z, axis = 1, keep_dims=True)
W2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))
# 定义输出节点
pred = tf.nn.softmax(tf.matmul(maxout, W2) + b2)  # Softmax 分类，正向结构
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z))
# 定义参数
learning_rate = 0.04
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 定义训练参数
training_epochs = 200  # 训练集样本迭代次数
batch_size = 100  # 训练过程中每次训练数据条数
display_step = 1  # 每训练一次显示具体的中间状态

# 启动 Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initializing OP
    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        # 循环所有数据
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # 计算平均 loss 值
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
    print("!!!Finished!!!Finished!!!Finished!!!Finished!!!")
    # 测试 Model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    # 保存模型
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    ####以下为读取并应用模型测试####
    sess.run(tf.global_variables_initializer())  # Initializing OP
    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    # 恢复模型变量
    saver.restore(sess, model_path)
    # 测试 Model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
