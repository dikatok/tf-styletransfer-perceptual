import tensorflow as tf


def instance_norm(x, name, epsilon=1e-5):
    with tf.variable_scope(name):
        gamma = tf.get_variable(shape=x.shape[-1], name="gamma")
        beta = tf.get_variable(shape=x.shape[-1], name="beta")
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name="norm")
    return x


def conv(inputs,
         name,
         filters,
         kernel_size,
         strides,
         with_bn=True,
         with_relu=True,
         data_format="channels_last"):
    """Convolution

    :param inputs: Input tensor
    :param name: Variable scope
    :param data_format: Either 'channels_last' or 'channels_first'
    :param filters: Filter size
    :param kernel_size: Kernel size
    :param strides: Strides
    :param with_bn: Either use instance normalization before non-linearity or not
    :param with_relu: Either use relu on output or not
    :return: Output tensor
    """

    padding = kernel_size // 2
    with tf.variable_scope(name):
        if data_format == "channels_last":
            paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        else:
            paddings = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        outputs = tf.pad(
            inputs,
            paddings=paddings,
            mode="REFLECT")
        outputs = tf.layers.conv2d(
            outputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            name="conv",
            data_format=data_format)
        if with_bn:
            outputs = instance_norm(outputs, name="inorm")
        if with_relu:
            outputs = tf.nn.relu(outputs, name="act")
    return outputs


def residual_block(inputs,
                   name,
                   filters,
                   kernel_size,
                   data_format="channels_last"):
    """Residual block

    :param inputs: Input tensor
    :param name: Variable scope
    :param data_format: Either 'channels_last' or 'channels_first'
    :param filters: Filter size
    :param kernel_size: Kernel size
    :return: Output tensor
    """

    with tf.variable_scope(name):
        residual = inputs
        outputs = conv(
            inputs,
            name="conv1",
            data_format=data_format,
            filters=filters,
            kernel_size=kernel_size,
            strides=1)
        outputs = conv(
            outputs,
            name="conv2",
            data_format=data_format,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            with_relu=False)
        outputs = tf.nn.relu(outputs + residual)
    return outputs


def upsample(inputs,
             name,
             data_format,
             filters,
             kernel_size,
             strides):
    """Upsample (resize -> conv) instead of transposed_conv or conv w/ 0.5 strides

    :param inputs: Input tensor
    :param name: Variable scope
    :param data_format: Either 'channels_last' or 'channels_first'
    :param filters: Filter size
    :param kernel_size: Kernel size
    :param strides: Strides
    :return: Output tensor
    """

    shape = inputs.shape.as_list()
    inferred_shape = tf.shape(inputs)

    spatial_axis = [1, 2] if data_format == "channels_last" else [2, 3]
    w, h = shape[spatial_axis[0]], shape[spatial_axis[1]]
    if w is None:
        w, h = inferred_shape[spatial_axis[0]], inferred_shape[spatial_axis[1]]

    with tf.variable_scope(name):
        if data_format == "channels_first":
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        outputs = tf.image.resize_images(inputs, size=[w * strides, h * strides])
        if data_format == "channels_first":
            outputs = tf.transpose(outputs, perm=[0, 3, 1, 2])
        outputs = conv(
            outputs,
            name="conv",
            data_format=data_format,
            filters=filters,
            kernel_size=kernel_size,
            strides=1)
    return outputs
