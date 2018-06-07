#!/usr/bin/python3
# 第四章  示例代码05  关于模型的内容、保存和载入

#1、保存模型
#之前是各种构建模型 graph 的操作（矩阵相乘、sigmoid 等）
saver = tf.train.Saver()     #生成 saver
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #先对模型初始化
    #然后将数据丢入模型进行训练
    #训练完成后，使用 saver.save 来保存
    saver.save(sess, "save_path/file_name")     #file_name如果不存在，会自动创建

#2、载入模型
saver = tf.train.Saver()     #生成 saver

with tf.Session() as sess:
    #参数可以进行初始化，也可以不进行初始化。若进行初始化，初始化的值会被 restore 的
    #值覆盖
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "save_path/file_name")     #会将已保存的变量值 restore 到变量中

#3、模型的内容
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "log/"
print_tensors_in_checkpoint_file(savedir + "file_name", None, True)

#4、保存模型的其他方法
#将变量 W 的值放到名字 weight 中
saver = tf.train.Saver({'weight': W, 'bias': b})
#或
saver = tf.train.Saver([W, b])  #放到 List 里
#或
saver = tf.train.Saver({v.op.name: v for v in [W, b]})   #将 op 的名字当作 key