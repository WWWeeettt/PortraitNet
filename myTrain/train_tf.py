import numpy as np
import argparse
import os
import torch
import tensorflow as tf
from tensorflow.python.framework import graph_util
from easydict import EasyDict as edict
from yaml import load

import sys
sys.path.append('../data/')
sys.path.append('../model/')

from datasets import Human


def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1 > 0] = 1
    sum2 = img + mask
    sum2[sum2 < 2] = 0
    sum2[sum2 >= 2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0*np.sum(sum2)/np.sum(sum1)


def main(args):
    print ('===========> loading config <============')
    config_path = args.config_path
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

    # set training dataset
    exp_args.istrain = True
    dataset_train = Human(exp_args)
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize,
                                                   shuffle=True, num_workers=args.workers)
    print ("image number in training: ", len(dataset_train))
    print ("finish load dataset ...")
    
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
    result = tf.nn.softmax(pred, name='result', dim=-1)
    
    with tf.variable_scope('loss'):
        softmaxs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred)
        softmaxs_loss = tf.reduce_mean(softmaxs, name="mean")

    gap = 0
    minloss =10000.0

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 20, 0.95, staircase=True)

    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(softmaxs_loss)

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
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state('./output/')
        # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['result'])

        for epoch in range(gap, 2000):

            print ('===========>   training    <==========={}/{}'.format(epoch+1, 2000))
            for i, (input_ori, input, edge, mask) in enumerate(dataLoader_train):
                input_x = input.cpu().detach().numpy()
                input_x = np.transpose(input_x, (0, 2, 3, 1))
                # input_y = mask.cpu().detach().numpy()[:, :, :, np.newaxis]
                input_y = mask.cpu().detach().numpy()
                _, loss_ = sess.run([optimizer, softmaxs_loss], {x: input_x, y: input_y})
                # print(loss_)
            if loss_ < minloss:
                saver.save(sess, './Model/result{}'.format(epoch))
                minloss = loss_
                print("minloss:", minloss)
        
        '''
            print ('===========>   testing    <===========')
            for i, (input_ori, input, edge, mask) in enumerate(dataLoader_test):
                input_x = input.cpu().detach().numpy()
                input_x = np.transpose(input_x, (0, 2, 3, 1))
                input_y = mask.cpu().detach().numpy()
                loss_ = sess.run(softmaxs_loss, {x: input_x, y: input_y})
                if i % args.printfreq == 0:
                    print('Step:', epoch, '| train loss: %.4f' % loss_)
        '''
        # with tf.gfile.FastGFile('portraitnet.pb', mode='wb') as f:
        #    f.write(output_graph_def.SerializeToString())
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--model', default='PortraitNet', type=str, 
                        help='<model> should in [PortraitNet, ENet, BiSeNet]')
    parser.add_argument('--config_path', 
                        default='/home/yupeng/Program/python/PortraitNet/config/model_mobilenetv2_without_auxiliary_losses.yaml',
                        type=str, help='the config path of the model')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--batchsize', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.1, type=float, help='momentum')
    parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--printfreq', default=100, type=int, help='print frequency')
    parser.add_argument('--savefreq', default=1000, type=int, help='save frequency')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    args = parser.parse_args()
    
    main(args)
