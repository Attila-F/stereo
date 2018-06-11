import os
import random
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from PIL import Image

# Torch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Custom
import config as cfg
from net import SiameseNet
from dataloader import KittiLoader
from loss import px3_loss

import numpy as np

# Stolen
def disparity_to_color(I):
    
    _map = np.array([[0,0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174], 
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]]
                   )      
    max_disp = 1.0*I.max()
    I = np.minimum(I/max_disp, np.ones_like(I))
    
    A = I.transpose()
    num_A = A.shape[0]*A.shape[1]
    
    bins = _map[0:_map.shape[0]-1,3]    
    cbins = np.cumsum(bins)    
    cbins_end = cbins[-1]
    bins = bins/(1.0*cbins_end)
    cbins = cbins[0:len(cbins)-1]/(1.0*cbins_end)
    
    A = A.reshape(1,num_A)            
    B = np.tile(A,(6,1))        
    C = np.tile(np.array(cbins).reshape(-1,1),(1,num_A))
       
    ind = np.minimum(sum(B > C),6)
    bins = 1/bins
    cbins = np.insert(cbins, 0,0)
    
    A = np.multiply(A-cbins[ind], bins[ind])   
    K1 = np.multiply(_map[ind,0:3], np.tile(1-A, (3,1)).T)
    K2 = np.multiply(_map[ind+1,0:3], np.tile(A, (3,1)).T)
    K3 = np.minimum(np.maximum(K1+K2,0),1)
    return np.reshape(K3, (I.shape[1],I.shape[0],3)).T

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    dataloader = KittiLoader(cfg.KITTIPATH, cfg.MAX_DISPARITY, cfg.KERNEL, cfg.CONV_LAYERS, cfg.BATCH_SIZE)
    model = SiameseNet(cfg.CHANNELS, cfg.FILTERS, cfg.MAX_DISPARITY, cfg.KERNEL, cfg.CONV_LAYERS, train_mode=False)
    model.load_state_dict(torch.load(cfg.SAVE_PATH.format(cfg.TEST_EPOCH)))
    model.eval()

    image_2_path = '/home/data_scene_flow/testing/image_2/000005_10.png'
    image_3_path = '/home/data_scene_flow/testing/image_3/000005_10.png'

    raw_image_2 = ndimage.imread(image_2_path, mode='RGB')
    raw_image_3 = ndimage.imread(image_3_path, mode='RGB')

    image_2 = torch.from_numpy(np.expand_dims(np.moveaxis((np.uint8(raw_image_2) - 128) / 256, -1, 0), 0))
    image_3 = torch.from_numpy(np.expand_dims(np.moveaxis((np.uint8(raw_image_3) - 128) / 256, -1, 0), 0))

    vec_2 = model(patch=Variable(image_2, requires_grad=False))
    vec_3 = model(patch=Variable(image_3, requires_grad=False))

    #pad = 2 * int(cfg.KERNEL / 2) * cfg.CONV_LAYERS
    output = torch.Tensor(cfg.MAX_DISPARITY, vec_2.size()[2], vec_2.size()[3], ).zero_() 

    for i in tqdm(range(cfg.MAX_DISPARITY)):
            slice_2 = vec_2[:,:,:,i:vec_2.size()[3]]
            slice_3 = vec_3[:,:,:,0:vec_2.size()[3]-i]
            inner_product = torch.sum(torch.mul(slice_2, slice_3), 1)
            output[i,:,i:vec_2.size()[3]] = inner_product.data.view(inner_product.data.size(1), inner_product.data.size(2))

    max_disp, max_disp_index = torch.max(output,0)
    max_disp_index = max_disp_index.view(output.size(1),output.size(2))

    # convert to colour
    color_map = disparity_to_color(max_disp_index.float().numpy())
    cmap = np.uint8(np.moveaxis(color_map, 0, 2)*128)
    print(cmap.shape)
    color_image = Image.fromarray(cmap, 'RGB')
    color_image.save('img_{}.png'.format(cfg.TEST_EPOCH))