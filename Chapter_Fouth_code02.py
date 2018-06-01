#!/usr/bin/python3
# 第四章  示例代码02  关于GPU资源使用


import tensorflow as tf

#指定GPU运算
# cpu:0  机器的CPU
# gpu:0  机器的第一个GPU
# gpu:1  机器的第二个GPU
"""
with tf.Session() as sess:
    with tf.device("/gpu:1"):
        a = tf.placeholder(tf.int16)
        ...
"""

#通过 tf.ConfigProto 构建一个 config ，在 config 中指定设备和参数，并在session中使用
# tf.ConfigProto 的参数
# log_device_placement = True ：是否打印设备分配日至
# allow_soft_placement = True ：如果指定的设备不存在，允许 TF 自动分配设备
"""
config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
session = tf.Session(config = config, ...)
"""

#设置按需分配资源
"""
config.gpu_options.allow_growth = True
"""

# config 创建时指定资源参数
"""
gpu_options = tf.GPUOptions(allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)   #分配给tensorflow的GPU显存为 70%
"""