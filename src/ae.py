import torch
from torch import nn
from torch.autograd import Variable
from torch import nn, distributions
import numpy as np
import copy
import pdb

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_z, channels, ff_shape, kernel, stride, padding, pool, conv_activation=None, ff_activation=None): 
        super().__init__()

        # conv_layers = []
        # batch_norm_layers = []
        # pool_layers = []
        ff_layers = []

        # C, W, H = dim_in
        # for ii in range(0,len(channels)-1):
        #     conv_layers.append(torch.nn.Conv2d(channels[ii], channels[ii+1], kernel[ii],
        #         stride=stride[ii], padding=padding[ii]))
        #     batch_norm_layers.append(torch.nn.BatchNorm2d(channels[ii+1]))
        #     # Keep track of output image size
        #     W = int(1+(W - kernel[ii] +2*padding[ii])/stride[ii])
        #     H = int(1+(H - kernel[ii] +2*padding[ii])/stride[ii])
        #     if pool[ii]:
        #         if W % pool[ii] != 0 or H % pool[ii] != 0:
        #             raise ValueError('trying to pool by non-factor')
        #         W, H = int(W/pool[ii]), int(H/pool[ii])
        #         pool_layers.append(torch.nn.MaxPool2d(pool[ii]))
        #     else:
        #         pool_layers.append(None)

        # self.cnn_output_size = W*H*channels[-1]
        # ff_shape = np.concatenate(([self.cnn_output_size], ff_shape))

        for ii in range(0, len(ff_shape) - 1):
          ff_layers.append(torch.nn.Linear(ff_shape[ii], ff_shape[ii+1]))
        ff_layers.append(torch.nn.Linear(ff_shape[ii], dim_z))

        self.dim_in = dim_in
        self.dim_out = dim_z 

        # self.conv_layers = torch.nn.ModuleList(conv_layers)
        # self.batch_norm_layers = torch.nn.ModuleList(batch_norm_layers)
        # if any(pool): 
        #     self.pool_layers = torch.nn.ModuleList(pool_layers)
        # else:
        #     self.pool_layers = pool_layers 
        
        self.ff_layers = torch.nn.ModuleList(ff_layers)

        # self.conv_activation = conv_activation
        self.ff_activation = ff_activation

    def forward(self, x):
        # # First compute convolutional pass
        # for ii in range(0,len(self.conv_layers)):
        #     x = self.conv_layers[ii](x)
        #     x = self.batch_norm_layers[ii](x)
        #     if self.conv_activation:
        #         x = self.conv_activation(x)
        #     if self.pool_layers[ii]:
        #         x = self.pool_layers[ii](x)
 
        # Flatten output and compress
        # x = x.view(x.shape[0], -1)
        for ii in range(0,len(self.ff_layers)):
            x = self.ff_layers[ii](x)
            if self.ff_activation:
                x = self.ff_activation(x)
        return x


def get_cartpole_encoder(dim_in, dim_z): 
    channels_enc = []
    ff_shape = [dim_in, 32, 32]

    conv_activation = None
    ff_activation = torch.nn.ReLU()

    n_channels = len(channels_enc) - 1
    kernel_enc = None 
    stride= None 
    padding= None 
    pool = None 

    return Encoder(dim_in, dim_z, channels_enc, ff_shape, kernel_enc, stride, padding, pool, conv_activation=conv_activation, ff_activation=ff_activation)

def compute_loss():
    #

    # Compute centroid
    centroids = torch.zeros((pp.n_strategies, dim_z))
    for ii in range(pp.n_strategies):
      centroids[ii] = torch.mean(encodings[enc_dict[ii]], axis=0)

    # Compute loss over batch

    return bound_loss.mean()/2, trans_loss.mean()
