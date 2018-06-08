#!/usr/bin/python3
# 第四章  示例代码08 获取检查点文件

ckpt = tf.train.get_checkpoint_state(ckpt_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
#或
kpt = tf.train.latest_checkpoint(savedir)
if kpt != None:
    saver.restore(sess, kpt)