# -*- coding: UTF-8 -*- 
# Authorized by Vlon Jang
# Created on 2017-09-26
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# ©2015-2017 All Rights Reserved.
#

"""
    Attention Model:
    WARNING: Use BatchNorm layer otherwise no accuracy gain.
    Lower layer with SpatialAttention, high layer with ChannelWiseAttention.
    In Visual155, Accuracy at 1, from 75.39% to 75.72%(↑0.33%).
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

def spatial_attention(feature_map, K=1024, weight_decay=0.00004, scope="", reuse=None):
    """This method is used to add spatial attention to model.
    
    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.
    
    Return
    ---------------
    @attended_fm: Feature map with Spatial Attention.
    """
    with tf.variable_scope(scope, 'SpatialAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
        w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b_s = tf.get_variable("SpatialAttention_b_s", [1],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        aa = tf.reshape(feature_map, [-1, C])  # (4,320,320,32) -> (409600,32)
        spatial_attention_fm = tf.matmul(aa, w_s) + b_s  #  -> (409600,1)
        bb = tf.reshape(spatial_attention_fm, [-1, W * H])  #  -> (4,102400)
        spatial_attention_fm = tf.nn.sigmoid(bb)  #  -> (4,102400)
#         spatial_attention_fm = tf.clip_by_value(tf.nn.relu(tf.reshape(spatial_attention_fm, 
#                                                                       [-1, W * H])), 
#                                                 clip_value_min = 0, 
#                                                 clip_value_max = 1)
        cc = tf.concat([spatial_attention_fm] * C, axis=1)  #  -> (4,3276800)
        attention = tf.reshape(cc, [-1, H, W, C])  #  -> (4,320,320,32)
        attended_fm = attention * feature_map
        return attended_fm

def spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
            assert max_pool.get_shape()[-1] == 1

            concat = tf.concat([avg_pool, max_pool], axis=3)
            assert concat.get_shape()[-1] == 2

            concat = slim.conv2d(concat, 1, kernel_size, padding='SAME', activation_fn=None, scope='conv')
            scale = tf.nn.sigmoid(concat)

            spatial_attention = scale * inputs

            return spatial_attention

def channel_wise_attention(feature_map, K=1024, weight_decay=0.00004, scope='', reuse=None):
    """This method is used to add spatial attention to model.
    
    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.
    
    Return
    ---------------
    @attended_fm: Feature map with Channel-Wise Attention.
    """
    with tf.variable_scope(scope, 'ChannelWiseAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
        w_s = tf.get_variable("ChannelWiseAttention_w_s", [C, C],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b_s = tf.get_variable("ChannelWiseAttention_b_s", [C],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        transpose_feature_map = tf.transpose(tf.reduce_mean(feature_map, [1, 2], keep_dims=True), 
                                             perm=[0, 3, 1, 2])
        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map, 
                                                         [-1, C]), w_s) + b_s
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)
#         channel_wise_attention_fm = tf.clip_by_value(tf.nn.relu(channel_wise_attention_fm), 
#                                                      clip_value_min = 0, 
#                                                      clip_value_max = 1)
        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (H * W), 
                                         axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map
        return attended_fm

if __name__ == "__main__":
    import os, sys
    # sys.path.append("/disk2/chunmeifeng/yzy/projects/Dual-OctMRI/code")
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    g=tf.Graph() #实例化一个类，用于 tensorflow 计算和表示用的数据流图
    with g.as_default():
        # i = tf.Variable(tf.random_uniform([4, 128, 128, 30]), name="var")
        i = tf.Variable(tf.random_uniform([4, 320, 320, 32]), name="var")
        a = spatial_attention(i)
        # model = tf.layers.conv2d(i, 1, (3, 3), padding='VALID')
        print(a.shape)