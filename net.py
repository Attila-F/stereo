import torch

from torch.nn import Module
from torch.nn import Sequential
from torch.nn import ReflectionPad2d
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import LogSoftmax
from torch.nn import ModuleList
from torch.nn.functional import relu

class SiameseNet(Module):    
    def __init__(self, channels, filters, max_disp, kernel, layers, train_mode=True):
        super(SiameseNet, self).__init__()                        
        self.max_disp = max_disp
        self.layers = []

        # First layer

        self.layers.append(Conv2d(channels, filters, kernel))
        self.layers.append(BatchNorm2d(filters, 1e-3))
        # Hidden layers
        for i in range(layers-1):
            self.layers.append(Conv2d(filters, filters, kernel))
            self.layers.append(BatchNorm2d(filters, 1e-3))
        self.softmax = LogSoftmax()

        self.layers = ModuleList(self.layers)

        self.train_mode = train_mode
    
    def base_forward(self, tensor):
        for i, layer in enumerate(self.layers):
            tensor = layer(tensor)
            # activate every 2nd layer except the last
            if i % 2 == 1 and i<len(self.layers)-1:
                tensor = relu(tensor)

        return tensor
    
    def top_forward(self, patch_2, patch_3):
        # Forward pass left sister
        feature_vec_2 = self.base_forward(patch_2)
        feature_vec_2 = feature_vec_2.view(feature_vec_2.size(0),1,64)

        # Forward pass right sister
        feature_vec_3 = self.base_forward(patch_3)
        feature_vec_3 = feature_vec_3.squeeze().view(feature_vec_3.size(0),64,self.max_disp+1)

        # Calculate inner product and softmax over disparities
        inner_product = feature_vec_2.bmm(feature_vec_3)
        #print(inner_product[0,:,:])
        logits = inner_product.view(patch_2.size(0),self.max_disp+1)

        
        return self.softmax(logits)

    def forward(self, **kwargs):
        if self.train_mode:
            return self.top_forward(kwargs['patch_2'], kwargs['patch_3'])
        else:
            return self.base_forward(kwargs['patch'])