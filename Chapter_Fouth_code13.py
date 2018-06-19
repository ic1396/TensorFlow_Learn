#!/usr/bin/python3
# 第四章  示例代码13 在作用域下使用 get_variable 以及嵌套 variable_scope 的用法

import tensorflow as tf

#作用域并列
#with tf.variable_scope("test1"):   #定义一个作用域 test1
#    var1 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)
#with tf.variable_scope("test2"):   #定义一个作用域 test2
#    var2 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)
#print("var1:", var1.name)
#print("var2:", var2.name)

#作用域嵌套
with tf.variable_scope("test1"):   #定义一个作用域 test1
    var1 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)
    with tf.variable_scope("test2"):   #定义一个作用域 test2
        var2 = tf.get_variable("firstvar", shape = [2], dtype = tf.float32)
print("var1:", var1.name)
print("var2:", var2.name)