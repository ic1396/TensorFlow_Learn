#!/usr/bin/python3
# 第三章　示例代码03  线性回归分析   不使用占位符

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize": [], "loss": []}  # 存放批次值和损失值

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]

# 生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
# 显示模拟数据
plt.plot(train_X, train_Y, 'ro', label='Original Data')
plt.legend()
plt.show()

# 创建模型

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(W, train_X) + b
# 反向优化
cost = tf.reduce_mean(tf.square(train_Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 梯度下降 Gradient descent

# 初始化所有变量
init = tf.global_variables_initializer()
# 定义训练参数
training_epochs = 20
display_step = 2

# 启动session
with tf.Session() as sess:
    sess.run(init)
    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost)
            print("Epoch: ", epoch + 1, "cost = ", loss, "W = ", sess.run(W), "b = ", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    print("cost = ", sess.run(cost), "W = ", sess.run(W), "b = ", sess.run(b))

    # 图形化训练结果
    plt.plot(train_X, train_Y, 'ro', label='Original Data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run VS. Training loss')

    plt.show()