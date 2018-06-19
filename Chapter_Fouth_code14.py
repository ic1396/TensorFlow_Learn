#!/usr/bin/python3
# 第四章  示例代码14 在作用域下使用 get_variable、reuse 属性以及嵌套 variable_scope 的用法

#使用作用域中的 reuse 参数来实现共享变量功能
import tensorflow as tf

tf.reset_default_graph()  #重置默认数据流图

with tf.variable_scope("test1"):   #定义一个作用域 test1
    var1 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)
    with tf.variable_scope("test2"):   #定义一个作用域 test2
        var2 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)

with tf.variable_scope("test1", reuse = True):   #定义一个作用域 test1
    var3 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)
    with tf.variable_scope("test2"):   #定义一个作用域 test2
        var4 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)
print("var3:", var3.name)
print("var4:", var4.name)