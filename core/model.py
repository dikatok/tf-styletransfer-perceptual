import tensorflow as tf

from core.layers import conv, residual_block, upsample


def create_model_fn(data_format="channels_last"):
    """Higher order function to create model_fn

    :param data_format: Either 'channels_last' or 'channels_first'
    :return: model_fn with signature as follows: inputs
    """

    def model_fn(inputs):
        with tf.variable_scope("tnet", reuse=tf.AUTO_REUSE):
            inputs = ((inputs / 255.) - 0.5) * 2
            conv1 = conv(inputs, name="conv1", data_format=data_format, filters=32, kernel_size=9, strides=1)
            conv2 = conv(conv1, name="conv2", data_format=data_format, filters=64, kernel_size=3, strides=2)
            conv3 = conv(conv2, name="conv3", data_format=data_format, filters=128, kernel_size=3, strides=2)
            res1 = residual_block(conv3, name="res1", data_format=data_format, filters=128, kernel_size=3)
            res2 = residual_block(res1, name="res2", data_format=data_format, filters=128, kernel_size=3)
            res3 = residual_block(res2, name="res3", data_format=data_format, filters=128, kernel_size=3)
            res4 = residual_block(res3, name="res4", data_format=data_format, filters=128, kernel_size=3)
            res5 = residual_block(res4, name="res5", data_format=data_format, filters=128, kernel_size=3)
            up1 = upsample(res5, name="up1", data_format=data_format,  filters=64, kernel_size=3, strides=2)
            up2 = upsample(up1, name="up2", data_format=data_format, filters=32, kernel_size=3, strides=2)
            conv4 = conv(
                up2,
                name="conv4",
                data_format=data_format,
                filters=3,
                kernel_size=9,
                strides=1,
                with_bn=False,
                with_relu=False)
            out = tf.clip_by_value(conv4, 0., 255.)
        return out

    return model_fn
