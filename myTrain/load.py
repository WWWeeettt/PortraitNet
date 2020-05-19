import os
import sys
import torch
import shutil
import argparse
import numpy as np
from yaml import load
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser
from mmdnn.conversion.pytorch.pytorch_graph import PytorchGraph

sys.path.append('../model/')
import model_mobilenetv2_seg_small as modellib


class MyParser(PytorchParser):
    def __init__(self, model_file_name, input_shape):
        super(PytorchParser, self).__init__()
        # if not os.path.exists(model_file_name):
        #     print("Pytorch model file [{}] is not found.".format(model_file_name))
        #     assert False
        # # test

        # # cpu: https://github.com/pytorch/pytorch/issues/5286
        # try:
        #     model = torch.load(model_file_name)
        # except:
        #     model = torch.load(model_file_name, map_location='cpu')
        model = model_file_name
        self.weight_loaded = True

        # Build network graph
        self.pytorch_graph = PytorchGraph(model)
        self.input_shape = tuple([1] + input_shape)
        self.pytorch_graph.build(self.input_shape)
        self.state_dict = self.pytorch_graph.state_dict
        self.shape_dict = self.pytorch_graph.shape_dict


def main(args):
    cudnn.benchmark = True
    assert args.model in ['PortraitNet', 'ENet', 'BiSeNet'], 'Error!, <model> should in [PortraitNet, ENet, BiSeNet]'
    
    print ('===========> loading config <============')
    config_path = args.config_path
    print ("config path: ", config_path)
    with open(config_path, 'rb') as f:
        cont = f.read()
    cf = load(cont)

    exp_args = edict()

    exp_args.model_root = cf['model_root']

    # set log path
    logs_path = os.path.join(exp_args.model_root, 'log/')
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)

    # if exp_args.video=True, add prior channel for input images, default=False
    exp_args.video = cf['video']

    # whether to add boundary auxiliary loss, default=False
    exp_args.addEdge = cf['addEdge']
    
    # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
    exp_args.useUpsample = cf['useUpsample'] 
    # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
    exp_args.useDeconvGroup = cf['useDeconvGroup'] 

    print ('===========> loading model <===========')

    netmodel = modellib.MobileNetV2(n_class=2,
                           useUpsample=exp_args.useUpsample,
                           useDeconvGroup=exp_args.useDeconvGroup,
                           addEdge=exp_args.addEdge,
                           channelRatio=1.0,
                           minChannel=16,
                           weightInit=True,
                           video=exp_args.video)
    print ("finish load PortraitNet ...")

    bestModelFile = os.path.join(exp_args.model_root, 'model_best.pth.tar')
    if os.path.isfile(bestModelFile):
        checkpoint = torch.load(bestModelFile)
        netmodel.load_state_dict(checkpoint['state_dict'])
        pytorchparser = MyParser(netmodel, [3, 224, 224])
        pytorchparser.run('model_best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--model', default='PortraitNet', type=str, 
                        help='<model> should in [PortraitNet, ENet, BiSeNet]')
    parser.add_argument('--config_path', 
                        default='/home/yupeng/Program/python/PortraitNet/config/model_mobilenetv2_without_auxiliary_losses.yaml',
                        type=str, help='the config path of the model')

    args = parser.parse_args()
    
    main(args)
