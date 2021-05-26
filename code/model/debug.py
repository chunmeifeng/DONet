import tensorflow as tf

def dense_block(in_data):
    pass

def channel_attention(in_data, r=16):
    """ Channel Attention Module
    """
    _, H, W, C = tuple([int(x) for x in in_data.get_shape()])
    ave_pool = tf.reduce_mean(in_data, [1, 2], keep_dims=True)  # (4, 1, 1, 32)
    fc1 = tf.layers.conv2d(ave_pool, C//r, (1,1), padding='SAME', activation=tf.nn.relu, use_bias=False)
    attention_weight = tf.layers.conv2d(fc1, C, (1,1), padding='SAME', activation=tf.nn.sigmoid, use_bias=False)
    out_data = in_data * attention_weight

    return out_data

def spatial_attention(in_data):
    """ Spatial Attention Module
    """
    # avg_pool = tf.reduce_mean(in_data, axis=3, keepdims=True)
    # assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.reduce_max(in_data, axis=3, keepdims=True)
    assert max_pool.get_shape()[-1] == 1

    # concat = tf.concat([avg_pool, max_pool], axis=3)
    # assert concat.get_shape()[-1] == 2

    attention_weight = tf.layers.conv2d(max_pool, 1, (7,7), padding='SAME', activation=tf.nn.sigmoid, use_bias=False)

    spatial_attention = in_data * attention_weight

    return spatial_attention


if __name__ == "__main__":
    import os, sys
    # sys.path.append("/disk2/chunmeifeng/yzy/projects/Dual-OctMRI/code")
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    g=tf.Graph() #实例化一个类，用于 tensorflow 计算和表示用的数据流图
    with g.as_default():
        # i = tf.Variable(tf.random_uniform([4, 128, 128, 30]), name="var")
        i = tf.Variable(tf.random_uniform([4, 320, 320, 32]), name="var")
        a = channel_attention(i)
        b = spatial_attention(i)
        # model = tf.layers.conv2d(i, 1, (3, 3), padding='VALID')
        print(a.shape)


        print('done')