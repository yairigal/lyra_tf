import tensorflow as tf

keras = tf.keras
from keras import models, layers
import numpy as np


class ResLayer(tf.keras.layers.Layer):
    def __init__(self, cin, cout, depth_dilation=1,last_conv_groups=1, **kwargs):
        super().__init__(**kwargs)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=(3, 1), dilation_rate=(depth_dilation, 1))
        self.pointwise = layers.Conv2D(filters=cin, kernel_size=(1, 1))
        self.leaky_relu = layers.LeakyReLU()
        self.last_conv = layers.Conv2D(filters=cout, kernel_size=(1, 1), groups=last_conv_groups)

    def call(self, res_input, inputs, **kwargs):
        x = inputs

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.leaky_relu(x)
        x = self.last_conv(x)
        return tf.add(res_input, x)


def apply_concat(conv_input, concat_shape, reversed=False):
    current_shape = conv_input.shape.as_list()
    current_shape[1] = concat_shape
    to_concat = tf.zeros(current_shape)
    if not reversed:
        conv_input = tf.concat([to_concat, conv_input], axis=1)
    else:
        conv_input = tf.concat([conv_input, to_concat], axis=1)
    return conv_input


class Encoder(tf.keras.layers.Layer):
    def __init__(self, cin, first_kernel_size=10, first_stride=5, concat_shapes=(2, 6, 18), first_concat=5, first_conv_groups=1, last_conv_groups=1, **kwargs):
        super().__init__(**kwargs)
        self.first_conv = layers.Conv2D(filters=cin, kernel_size=(first_kernel_size, 1), strides=(first_stride, 1), groups=first_conv_groups)
        self.res1 = ResLayer(cin=cin, cout=cin, depth_dilation=1, last_conv_groups=last_conv_groups)
        self.res2 = ResLayer(cin=cin, cout=cin, depth_dilation=3, last_conv_groups=last_conv_groups)
        self.res3 = ResLayer(cin=cin, cout=cin, depth_dilation=9, last_conv_groups=last_conv_groups)
        self.concat_shapes = concat_shapes
        self.first_concat = first_concat

    def call(self, inputs, *args, **kwargs):
        x = apply_concat(inputs, self.first_concat)
        res_inputs = self.first_conv(x)
        conv_input = tf.nn.leaky_relu(res_inputs)

        conv_input = apply_concat(conv_input, self.concat_shapes[0])

        x = self.res1(res_inputs, conv_input)
        res_inputs = tf.add(x, res_inputs)
        conv_input = tf.nn.leaky_relu(res_inputs)

        conv_input = apply_concat(conv_input, self.concat_shapes[1])

        x = self.res2(res_inputs, conv_input)
        res_inputs = tf.add(x, res_inputs)
        conv_input = tf.nn.leaky_relu(res_inputs)

        conv_input = apply_concat(conv_input, self.concat_shapes[2])

        x = self.res3(res_inputs, conv_input)
        res_inputs = tf.add(x, res_inputs)
        x = tf.nn.leaky_relu(res_inputs)
        return x


class SoundStreamEncoder(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc1 = Encoder(cin=64, first_kernel_size=64, first_stride=16, concat_shapes=[2, 6, 18], first_concat=48, last_conv_groups=1)
        self.enc2 = Encoder(cin=128, first_kernel_size=10, first_stride=5, concat_shapes=[2, 6, 18], first_concat=5, last_conv_groups=2)
        self.enc3 = Encoder(cin=256, first_kernel_size=4, first_stride=2, concat_shapes=[2, 6, 18], first_concat=2, last_conv_groups=4, first_conv_groups=2)
        self.conv1 = layers.Conv2D(filters=512, kernel_size=(4, 1), strides=(2, 1), groups=4)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 1), groups=4)

    def call(self, inputs, training=None, mask=None):
        x = self.enc1(inputs)
        x = self.enc2(x)
        x = self.enc3(x)
        x = apply_concat(x, 2, reversed=False)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)
        x = apply_concat(x, 2)
        x = self.conv2(x)
        x = tf.squeeze(x, 1)
        return x

from itertools import permutations
def try_set_weights(root_obj, weights, bias):

    root_obj.set_weights([
        weights.transpose((1, 2, 3, 0)),
        bias
    ])
    return


    transposes = [
        # (3, 2, 1, 0),
        # (1, 0, 3, 2),
        # (2, 1, 0, 3),
        # (0, 3, 2, 1),
        # (3, 0, 1, 2),
        (1, 2, 3, 0)
    ]

    for tran in transposes:
        error = None
        try:
            root_obj.set_weights([
                weights.transpose(tran),
                bias
            ])
            # print(f'found match {root_obj} -> {tran}')
            return

        except ValueError as e:
            error = e
            continue

    print(root_obj)
    print(weights.shape)
    print(root_obj.weights[0].shape)
    raise error


def load_weights(model):
    for enc in ['enc1',
                'enc2',
                'enc3',
                ]:
        try_set_weights(getattr(model, enc).first_conv,
                        np.load(f'weights\\{enc}.first_conv.weights.npy'),
                        np.load(f'weights\\{enc}.first_conv.bias.npy'))

        for res in ['res1', 'res2', 'res3']:
            current_enc = getattr(model, enc)
            current_res = getattr(current_enc, res)

            # weight = np.load(f'weights\\{enc}.{res}.depthwise.weights.npy')
            # print(weight.max(), weight.min())
            #
            # weight = np.load(f'weights\\{enc}.{res}.pointwise.weights.npy')
            # print(weight.max(), weight.min())
            #
            # weight = np.load(f'weights\\{enc}.{res}.last_conv.weights.npy')
            # print(weight.max(), weight.min())

            try_set_weights(current_res.depthwise,
                            np.load(f'weights\\{enc}.{res}.depthwise.weights.npy'),
                            np.load(f'weights\\{enc}.{res}.depthwise.bias.npy'))
            try_set_weights(current_res.pointwise,
                            np.load(f'weights\\{enc}.{res}.pointwise.weights.npy'),
                            np.load(f'weights\\{enc}.{res}.pointwise.bias.npy'))
            try_set_weights(current_res.last_conv,
                            np.load(f'weights\\{enc}.{res}.last_conv.weights.npy'),
                            np.load(f'weights\\{enc}.{res}.last_conv.bias.npy'))

    try_set_weights(model.conv1,
                    np.load(f'weights\\conv1.weights.npy'),
                    np.load(f'weights\\conv1.bias.npy'), )
    try_set_weights(model.conv2,
                    np.load(f'weights\\conv2.weights.npy'),
                    np.load(f'weights\\conv2.bias.npy'), )
    return model


def test_model(model):
    inter = tf.lite.Interpreter('./soundstream_encoder.tflite')
    inter.allocate_tensors()
    tflite_model = inter.get_signature_runner()
    x = np.float32(np.random.random((1, 320)))
    res = tflite_model(input_audio=x)['output_0']

    my_res = model(x.reshape(1, 320, 1, 1))

    print(np.allclose(res, my_res))


    import ipdb;
    ipdb.set_trace(context=20)


if __name__ == '__main__':
    # block(tf.zeros([1, 2, 3, 3]))

    # block = layers.Conv2D(filters=64, kernel_size=(64, 1), strides=(16, 1))
    # res = ResLayer(cin=64, cout=64, concat_shape=2)
    # block.build([1, 368, 1, 1])
    # res.build(([1, 20, 1, 64], [1, 22, 1, 64]))
    # result = block(tf.zeros([1, 368, 1, 1]))
    # # depth_in = tf.concat([tf.zeros([1, 2, 1, 64]), result], axis=1)
    # result = res(result, depth_in)
    model = SoundStreamEncoder()
    model.build([1, 320, 1, 1])
    # model.summary()
    res = model(tf.zeros([1, 320, 1, 1]))

    model = load_weights(model)

    test_model(model)

    # model = SoundstreamEncoder
    # model.build((1, 368, 1, 1))
    # model.summary()
