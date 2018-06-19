#!/usr/bin/python3
# 第四章  示例代码15 共享变量的作用域与初始化

import tensorflow as tf

tf.reset_default_graph()  # 重置默认数据流图

with tf.variable_scope("test1", initializer=tf.constant_initializer(0.4)):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
        var3 = tf.get_variable("var3", shape=[2], initializer=tf.constant_initializer(0.3))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1:", var1.eval())  # 作用域 test1 下的变量
    print("var2:", var2.eval())  # 作用域 test2 下的变量，继承 test1 的初始化
    print("var3:", var3.eval())  # 作用域 test2 下的变量