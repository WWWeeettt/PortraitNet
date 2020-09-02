import tensorflow as tf
import numpy as np

def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # print filt
    filt_tf = tf.convert_to_tensor(filt)
    w = tf.zeros(shape=[num_channels, 1, size, size])
    for i in range(num_channels):
        w[i, 0] = filt_tf
    return w

def conv_bn(x, oup, kernel, stride):
    conv1 = tf.layers.conv2d(x, oup, kernel, strides=stride, padding='SAME', use_bias=False)
    bn1 = tf.layers.batch_normalization(conv1, momentum=0.9, epsilon=1e-05)
    return tf.nn.relu(bn1)

def conv_dw(x, inp, oup, kernel, stride):
    conv1 = tf.layers.conv2d(x, inp, kernel, strides=stride, padding='SAME', use_bias=False)
    bn1 = tf.layers.batch_normalization(conv1, momentum=0.9, epsilon=1e-05)
    h1 = tf.nn.relu(bn1)
    
    conv2 = tf.layers.conv2d(h1, oup, 1, strides=1, padding='SAME', use_bias=False)
    bn2 = tf.layers.batch_normalization(conv2, momentum=0.9, epsilon=1e-05)
    return tf.nn.relu(bn2)

def InvertedResidual(x, inp, oup, stride, expand_ratio, dilation=1):
    assert stride in [1, 2]
    use_res_connect = (stride == 1 and inp == oup)

    # pw
    conv1 = tf.layers.conv2d(x, inp*expand_ratio, 1, strides=1, padding='SAME', dilation_rate=1, use_bias=False)
    bn1 = tf.layers.batch_normalization(conv1, momentum=0.9, epsilon=1e-05)
    h1 = tf.nn.relu(bn1)
    # dw
    conv2 = tf.layers.conv2d(h1, inp*expand_ratio, 3, strides=stride, padding='SAME', dilation_rate=dilation, use_bias=False)
    bn2 = tf.layers.batch_normalization(conv2, momentum=0.9, epsilon=1e-05)
    h2 = tf.nn.relu(bn2)
    # pw-linear
    conv3 = tf.layers.conv2d(h2, oup, 1, strides=1, padding='SAME', dilation_rate=1, use_bias=False)
    bn3 = tf.layers.batch_normalization(conv3, momentum=0.9, epsilon=1e-05)
    
    if use_res_connect:
        return x + bn3
    else:
        return bn3
    
# Residual Block
def ResidualBlock(x, inp, oup, stride=1):
    conv1 = conv_dw(x, inp, oup, 3, stride)
    conv2 = tf.layers.conv2d(conv1, oup, 3, strides=1, padding='SAME', use_bias=False)
    bn2 = tf.layers.batch_normalization(conv2, momentum=0.9, epsilon=1e-05)
    h2 = tf.nn.relu(bn2)
    conv3 = tf.layers.conv2d(h2, oup, 1, strides=1, padding='SAME', use_bias=False)
    block = tf.layers.batch_normalization(conv3, momentum=0.9, epsilon=1e-05)
    
    if inp == oup:
        residual = x
    else:
        conv4 = tf.layers.conv2d(x, oup, 1, strides=1, padding='SAME', use_bias=False)
        residual = tf.layers.batch_normalization(conv4, momentum=0.9, epsilon=1e-05)

    return tf.nn.relu(block + residual)

class MobileNetV2():
    def __init__(self, n_class=2, addEdge=False,
                 channelRatio=1.0, minChannel=16):
        '''
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        '''
        self.addEdge = addEdge
        self.channelRatio = channelRatio
        self.minChannel = minChannel
        self.n_class = n_class

    def build(self, x):
        with tf.variable_scope('stage0'):
            self.stage0 = conv_bn(x, self.depth(8), 3, 2)
        with tf.variable_scope('stage1'):
            self.stage1 = InvertedResidual(self.stage0, self.depth(8), self.depth(4), 1, 1) # 1/2

        # with tf.variable_scope('ir1'):
        #     ir1 = InvertedResidual(self.stage1, self.depth(16), self.depth(24), 2, 6)
        with tf.variable_scope('stage2'):
            self.stage2 = InvertedResidual(self.stage1, self.depth(4), self.depth(6), 2, 6)

        # with tf.variable_scope('ir2'):
        #     ir2 = InvertedResidual(self.stage2, self.depth(24), self.depth(32), 2, 6)
        # with tf.variable_scope('ir3'):
        #     ir3 = InvertedResidual(ir2, self.depth(32), self.depth(32), 1, 6)
        with tf.variable_scope('stage3'):
            self.stage3 = InvertedResidual(self.stage2, self.depth(6), self.depth(8), 2, 6)

        # with tf.variable_scope('ir4'):
        #     ir4 = InvertedResidual(self.stage3, self.depth(32), self.depth(64), 2, 6)
        # with tf.variable_scope('ir5'):
        #     ir5 = InvertedResidual(ir4, self.depth(64), self.depth(64), 1, 6)
        with tf.variable_scope('ir1'):
            ir1 = InvertedResidual(self.stage3, self.depth(8), self.depth(16), 2, 6)
        with tf.variable_scope('stage4'):
            self.stage4 = InvertedResidual(ir1, self.depth(16), self.depth(16), 1, 6)

        # with tf.variable_scope('ir7'):
        #     ir7 = InvertedResidual(self.stage4, self.depth(64), self.depth(96), 1, 6)
        # with tf.variable_scope('ir8'):
        #     ir8 = InvertedResidual(ir7, self.depth(96), self.depth(96), 1, 6)
        with tf.variable_scope('stage5'):
            self.stage5 = InvertedResidual(self.stage4, self.depth(16), self.depth(24), 1, 6)

        # with tf.variable_scope('ir9'):
        #     ir9 = InvertedResidual(self.stage5, self.depth(96), self.depth(160), 2, 6)
        # with tf.variable_scope('ir10'):
        #     ir10 = InvertedResidual(ir9, self.depth(160), self.depth(160), 1, 6)
        with tf.variable_scope('stage6'):
            self.stage6 = InvertedResidual(self.stage5, self.depth(24), self.depth(40), 2, 6)
        
        with tf.variable_scope('stage7'):
            self.stage7 = InvertedResidual(self.stage6, self.depth(40), self.depth(80), 1, 6) # 1/32

        with tf.variable_scope('transit1'):
            self.transit1 = ResidualBlock(self.stage7, self.depth(80), self.depth(24))
        self.deconv1 = tf.layers.conv2d_transpose(self.transit1, self.depth(24), 4, strides=2, padding='same',
                                                  use_bias=False, name='deconv1')

        with tf.variable_scope('transit2'):
            self.transit2 = ResidualBlock((self.stage5+self.deconv1), self.depth(24), self.depth(8))
        self.deconv2 = tf.layers.conv2d_transpose(self.transit2, self.depth(8), 4, strides=2, padding='same',
                                                  use_bias=False, name='deconv2')

        with tf.variable_scope('transit3'):
            self.transit3 = ResidualBlock((self.stage3+self.deconv2), self.depth(8), self.depth(6))
        self.deconv3 = tf.layers.conv2d_transpose(self.transit3, self.depth(6), 4, strides=2, padding='same',
                                                  use_bias=False, name='deconv3')

        with tf.variable_scope('transit4'):
            self.transit4 = ResidualBlock((self.stage2+self.deconv3), self.depth(6), self.depth(4))
        self.deconv4 = tf.layers.conv2d_transpose(self.transit4, self.depth(4), 4, strides=2, padding='same',
                                                  use_bias=False, name='deconv4')

        with tf.variable_scope('transit5'):
            self.transit5 = ResidualBlock(self.deconv4, self.depth(4), self.depth(2))
        self.deconv5 = tf.layers.conv2d_transpose(self.transit5, self.depth(2), 4, strides=2, padding='same',
                                                  use_bias=False, name='deconv5')

        with tf.variable_scope('Output'):
            self.pred = tf.layers.conv2d(self.deconv5, self.n_class, 3, strides=1, padding='SAME', use_bias=False, name='pred')
        
        # print("stage0", self.stage0.shape)
        # print("stage1", self.stage1.shape)
        # print("ir1", ir1.shape)
        # print("stage2", self.stage2.shape)
        # print("ir2", ir2.shape)
        # print("ir3", ir3.shape)
        # print("stage3", self.stage3.shape)
        # print("ir4", ir4.shape)
        # print("ir5", ir5.shape)
        # print("ir6", ir6.shape)
        # print("stage4", self.stage4.shape)
        # print("ir7", ir7.shape)
        # print("ir8", ir8.shape)
        # print("stage5", self.stage5.shape)
        # print("ir9", ir9.shape)
        # print("ir10", ir10.shape)
        # print("stage6", self.stage6.shape)
        # print("stage7", self.stage7.shape)
        # print("transit1", self.transit1.shape)
        # print("transit2", self.transit2.shape)
        # print("transit3", self.transit3.shape)
        # print("transit4", self.transit4.shape)
        # print("transit5", self.transit5.shape)
        # print("pred", self.pred.shape)

        # if self.addEdge:
        #     self.edge = tf.layers.conv2d(self.deconv5, self.n_class, 3, strides=1, padding='SAME', use_bias=False)
        #     return self.pred, self.edge
        # else:
        return self.pred

    def __call__(self, *args, **kwargs):
        # if self.addEdge:
        #     return self.pred, self.edge
        # else:
        return self.pred

    def depth(self, channels):
        min_channel = min(channels, self.minChannel)
        return max(min_channel, int(channels*self.channelRatio))
