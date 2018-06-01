#!/usr/bin/python3
# 第四章  示例代码01 session的使用

import tensorflow as tf

#session HelloWorld
"""
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
sess.close()
"""

#使用 with session 计算常量
"""
a = tf.constant(3)
b = tf.constant(4)
with tf.Session() as sess:
    print("加法 a + b = %i" % sess.run(a + b))
    print("乘法 a * b = %i" % sess.run(a * b))
    print("加法 a + b = ", sess.run(a + b))
    print("乘法 a * b = ", sess.run(a * b))
"""
#数据注入机制
"""
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    #计算具体数值
    print("加法 a + b = %i" % sess.run(add, feed_dict = {a: 3, b: 4}))
    print("乘法 a * b = %i" % sess.run(mul, feed_dict = {a: 3, b: 4}))
    print("加法 a + b =", sess.run(add, feed_dict = {a: 3, b: 4}))
    print("乘法 a * b =", sess.run(mul, feed_dict = {a: 3, b: 4}))
"""
#交互式 session
"""
sess = tf.InteractiveSession()
"""

#使用 Supervisor 方式

#使用注入机制获取节点
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    #计算具体数值
    #将op通过run打印出来
    print("加法 a + b = %i" % sess.run(add, feed_dict = {a: 3, b: 4}))
    print("乘法 a * b = %i" % sess.run(mul, feed_dict = {a: 3, b: 4}))
    print("加法 a + b =", sess.run(add, feed_dict = {a: 3, b: 4}))
    print("乘法 a * b =", sess.run(mul, feed_dict = {a: 3, b: 4}))
    #将节点打印出来
    print(sess.run([mul, add], feed_dict = {a: 5, b: 6}))