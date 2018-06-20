#!/usr/bin/python3
# 第四章  示例代码16 variable_scope 的 as 用法，以及对应的作用域

import tensorflow as tf

tf.reset_default_graph()  # 为方便在 spyder 中执行，重置默认数据流图，在其他环境下可视情况删除

with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v", [1])

print("sp: ", sp.name)  # 作用域名称
print("var1: ", var1.name)

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])

    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])

        with tf.variable_scope(""):
            var4 = tf.get_variable("v4", [1])

print("sp1: ", sp1.name)
print("var2: ", var2.name)
print("var3: ", var3.name)

with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])  # v 为一个变量
        x = 1.0 + v  # x 为一个op，实现 1.0 + v 操作
        with tf.name_scope(""):
            y = 1.0 + v

print("v: ", v.name)
print("x.op: ", x.op.name)

print("var4: ", var4.name)
print("y.op: ", y.op.name)

