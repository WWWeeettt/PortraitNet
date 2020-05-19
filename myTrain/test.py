import numpy as np
import os
import cv2
import torch
import tensorflow as tf
from tensorflow.python.framework import graph_util
from easydict import EasyDict as edict
from yaml import load

import sys
sys.path.append('../data/')
sys.path.append('../model/')

from datasets import Human
from data_aug import Normalize_Img, Anti_Normalize_Img


def generate_input(exp_args, inputs, prior=None):
    inputs_norm = Normalize_Img(inputs, scale=exp_args.img_scale, mean=exp_args.img_mean, val=exp_args.img_val)

    if exp_args.video:
        if prior is None:
            prior = np.zeros((exp_args.input_height, exp_args.input_width, 1))
            inputs_norm = np.c_[inputs_norm, prior]
        else:
            prior = prior.reshape(exp_args.input_height, exp_args.input_width, 1)
            inputs_norm = np.c_[inputs_norm, prior]

    return np.array(inputs_norm, dtype=np.float32)

def main():
    print ('===========> loading config <============')
    config_path = '/home/yupeng/Program/python/PortraitNet/config/model_mobilenetv2_without_auxiliary_losses.yaml'
    print ("config path: ", config_path)
    with open(config_path, 'rb') as f:
        cont = f.read()
    cf = load(cont)

    exp_args = edict()

    exp_args.istrain = cf['istrain']  # set the mode
    exp_args.task = cf['task']  # only support 'seg' now
    exp_args.datasetlist = cf['datasetlist']
    exp_args.model_root = cf['model_root']
    exp_args.data_root = cf['data_root']
    exp_args.file_root = cf['file_root']

    # the height of input images, default=224
    exp_args.input_height = cf['input_height']
    # the width of input images, default=224
    exp_args.input_width = cf['input_width']

    # if exp_args.video=True, add prior channel for input images, default=False
    exp_args.video = cf['video']
    # the probability to set empty prior channel, default=0.5
    exp_args.prior_prob = cf['prior_prob']

    # whether to add boundary auxiliary loss, default=False
    exp_args.addEdge = cf['addEdge']
    # the weight of boundary auxiliary loss, default=0.1
    exp_args.edgeRatio = cf['edgeRatio']
    # whether to add consistency constraint loss, default=False
    exp_args.stability = cf['stability']
    # whether to use KL loss in consistency constraint loss, default=True
    exp_args.use_kl = cf['use_kl']
    # temperature in consistency constraint loss, default=1
    exp_args.temperature = cf['temperature']
    # the weight of consistency constraint loss, default=2
    exp_args.alpha = cf['alpha']

    # input normalization parameters
    exp_args.padding_color = cf['padding_color']
    exp_args.img_scale = cf['img_scale']
    # BGR order, image mean, default=[103.94, 116.78, 123.68]
    exp_args.img_mean = cf['img_mean']
    # BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
    exp_args.img_val = cf['img_val']

    # whether to use pretian model to init portraitnet
    exp_args.init = cf['init']
    # whether to continue training
    exp_args.resume = cf['resume']

    # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
    exp_args.useUpsample = cf['useUpsample']
    # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
    exp_args.useDeconvGroup = cf['useDeconvGroup']

    print ('===========> loading data <===========')
    # set testing dataset
    exp_args.istrain = False
    dataset_test = Human(exp_args)
    dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize,
                                                  shuffle=True, num_workers=args.workers)
    print ("image number in testing: ", len(dataset_test))

    '''
    print ('===========> loading model <===========')
    # train our model: portraitnet
    
    import model_mobilenetv2_seg_small_tf as modellib
    netmodel = modellib.MobileNetV2(n_class=2,
                                    addEdge=exp_args.addEdge,
                                    channelRatio=1.0,
                                    minChannel=16)
    print ("finish load PortraitNet ...")
    
    with tf.variable_scope('Inputs'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='x_input')
        y = tf.placeholder(dtype=tf.int64, shape=[None, 224, 224], name='y_input')

    pred = netmodel.build(x)
    #result = tf.nn.softmax(pred, name='result')
    
    with tf.variable_scope('loss'):
        softmaxs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred)
        softmaxs_loss = tf.reduce_mean(softmaxs)

    gap = 0

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 20, 0.95, staircase=True)

    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(softmaxs_loss)
    '''

    img_ori = cv2.imread("/home/yupeng/Program/python/Data/EG1800/Images/00457.png")

    with tf.Session() as sess:
        '''
        with tf.gfile.FastGFile('model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter("logs", tf.get_default_graph())
        writer.close()

        '''
        saver = tf.train.import_meta_graph('Model/model.meta')
        saver.restore(sess, "Model/model")

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("Inputs/x_input:0")
        y = graph.get_tensor_by_name("Inputs/y_input:0")
        pred = graph.get_tensor_by_name("pred:0")

        print ('===========>   testing    <===========')
        for i, (input_ori, input, edge, mask) in enumerate(dataLoader_test):
            input_x = input.cpu().detach().numpy()
            input_x = np.transpose(input_x, (0, 2, 3, 1))
            result = sess.run(pred, {x: input_x})
            softmaxs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=result)
            softmaxs_loss = tf.reduce_mean(softmaxs)
            print(softmaxs_loss)

        # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['result'])

        # with tf.gfile.FastGFile('portraitnet.pb', mode='wb') as f:
        #    f.write(output_graph_def.SerializeToString())

if __name__ == "__main__":
    main()