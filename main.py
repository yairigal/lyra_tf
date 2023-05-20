import tensorflow as tf

keras = tf.keras
from keras import models, layers
import numpy as np
import ipdb


class ResLayer(tf.keras.layers.Layer):
    def __init__(self, cin, cout, depth_dilation=1, last_conv_groups=1, **kwargs):
        super().__init__(**kwargs)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=(3, 1), dilation_rate=(depth_dilation, 1))
        self.pointwise = layers.Conv2D(filters=cin, kernel_size=(1, 1))
        self.leaky_relu = layers.LeakyReLU()
        self.last_conv = layers.Conv2D(filters=cout, kernel_size=(1, 1), groups=last_conv_groups)

    def call(self, inputs, **kwargs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.leaky_relu(x)
        x = self.last_conv(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, cin, first_concat, first_kernel_size=10, first_stride=5, first_conv_groups=1, last_conv_groups=1, **kwargs):
        super().__init__(**kwargs)
        self.first_conv = layers.Conv2D(filters=cin, kernel_size=(first_kernel_size, 1), strides=(first_stride, 1), groups=first_conv_groups)
        self.res1 = ResLayer(cin=cin, cout=cin, depth_dilation=1, last_conv_groups=last_conv_groups)
        self.res2 = ResLayer(cin=cin, cout=cin, depth_dilation=3, last_conv_groups=last_conv_groups)
        self.res3 = ResLayer(cin=cin, cout=cin, depth_dilation=9, last_conv_groups=last_conv_groups)
        self.first_concat = first_concat
        self.first_buffer = tf.zeros(first_concat)
        self.res1_buffer = tf.zeros((1, 2, 1, cin))
        self.res2_buffer = tf.zeros((1, 6, 1, cin))
        self.res3_buffer = tf.zeros((1, 18, 1, cin))

    @tf.function
    def _causal_pad(self, x, buffer, size):
        x = tf.concat([buffer, x], axis=1)
        next_padding = x[:, -size:, :, :]
        return x, next_padding

    def call(self, inputs, *args, **kwargs):
        x, self.first_buffer = self._causal_pad(inputs, self.first_buffer, tf.constant(self.first_concat[1]))
        # x = tf.concat([self.first_buffer, inputs], axis=1)
        # d = tf.Variable([0, -48, 0, 0], dtype=tf.int32)
        # e = tf.Variable([0, 0, 0, 1], dtype=tf.int32)
        # abc = tf.strided_slice(x, d, e)
        # self.first_buffer = x[:, tf.constant(-self.first_concat[1]):, :, :]
        res_inputs = self.first_conv(x)
        conv_input = tf.nn.leaky_relu(res_inputs)

        conv_input = tf.concat([self.res1_buffer, conv_input], axis=1)
        self.res1_buffer = conv_input[:, -2:, :, :]

        res_output = self.res1(conv_input)
        res_inputs = tf.add(res_inputs, res_output)
        conv_input = tf.nn.leaky_relu(res_inputs)

        conv_input = tf.concat([self.res2_buffer, conv_input], axis=1)
        self.res2_buffer = conv_input[:, -6:, :, :]

        res_outputs = self.res2(conv_input)
        res_inputs = tf.add(res_outputs, res_inputs)
        conv_input = tf.nn.leaky_relu(res_inputs)

        conv_input = tf.concat([self.res3_buffer, conv_input], axis=1)
        self.res3_buffer = conv_input[:, -18:, :, :]

        res_outputs = self.res3(conv_input)
        res_inputs = tf.add(res_outputs, res_inputs)
        x = tf.nn.leaky_relu(res_inputs)
        return x


class SoundStreamEncoder(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc1 = Encoder(cin=64, first_kernel_size=64, first_stride=16, first_concat=(1, 48, 1, 1), last_conv_groups=1)
        self.enc2 = Encoder(cin=128, first_kernel_size=10, first_stride=5, first_concat=(1, 5, 1, 64), last_conv_groups=2)
        self.enc3 = Encoder(cin=256, first_kernel_size=4, first_stride=2, first_concat=(1, 2, 1, 128), last_conv_groups=4, first_conv_groups=2)
        self.conv1 = layers.Conv2D(filters=512, kernel_size=(4, 1), strides=(2, 1), groups=4)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 1), groups=4)

        self.first_buffer = tf.zeros((1, 2, 1, 256))
        self.second_buffer = tf.zeros((1, 2, 1, 512))

    def call(self, inputs, training=None, mask=None):
        x = self.enc1(inputs)
        x = self.enc2(x)
        x = self.enc3(x)
        x = tf.concat([self.first_buffer, x], axis=1)
        self.first_buffer = x[:, -2:, :, :]
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)
        x = tf.concat([self.second_buffer, x], axis=1)
        self.second_buffer = x[:, -2:, :, :]
        x = self.conv2(x)
        x = tf.squeeze(x, 1)
        return x


# model = keras.models.Sequential([
#     Encoder(cin=64, first_kernel_size=64, first_stride=16, concat_shapes=[2, 6, 18], first_concat=48, last_conv_groups=1),
#     Encoder(cin=128, first_kernel_size=10, first_stride=5, concat_shapes=[2, 6, 18], first_concat=5, last_conv_groups=2),
#     Encoder(cin=256, first_kernel_size=4, first_stride=2, concat_shapes=[2, 6, 18], first_concat=2, last_conv_groups=4, first_conv_groups=2),
#     Concat(2),
#     layers.Conv2D(filters=512, kernel_size=(4, 1), strides=(2, 1), groups=4),
#     layers.LeakyReLU(),
#     Concat(2),
#     layers.Conv2D(filters=64, kernel_size=(3, 1), groups=4),
# ])


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

    print(my_res)
    print(res)
    print(np.allclose(res, my_res))
    # import ipdb;
    # ipdb.set_trace(context=20)


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
    ipdb.set_trace(context=20)

    model = load_weights(model)
    tflitemodel = tf.lite.TFLiteConverter.from_keras_model(model)
    tflitemodel = tflitemodel.convert()
    with open('my_model.tflite', 'wb') as f:
        f.write(tflitemodel)
    # print(tflitemodel)

    # tf.keras.models.save_model(model, './my_model.tf', save_format='keras')
    test_model(model)

    # model = SoundstreamEncoder
    # model.build((1, 368, 1, 1))
    # model.summary()
