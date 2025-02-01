""""
    We use some layers at pre-trained vgg19 model
    Content representation : Conv4_2
    Style representation   : Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv5_1

    forward
        abstract specific outputs written above by mode and save at results list during the vgg19 model
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19

layer_index = {
    "Conv4_2" : 21,
    "Conv1_1" : 0,
    "Conv2_1" : 5,
    "Conv3_1" : 10,
    "Conv4_1" : 19,
    "Conv5_1" : 28
}

class StyleTransfer(nn.Module):
    def __init__(self,):
        super(StyleTransfer, self).__init__()
        self.vgg19_model = vgg19(weights = 'DEFAULT')
        self.vgg19_features = self.vgg19_model.features

        self.content_layer = [layer_index["Conv4_2"]]
        self.style_layer = [layer_index["Conv1_1"], layer_index["Conv2_1"], layer_index["Conv3_1"], layer_index["Conv4_1"], layer_index["Conv5_1"]]

    def forward(self, x, mode = None):
        results = []

        if mode == "Content":
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.content_layer:
                    results.append(x)
        
        elif mode == "Style":
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.style_layer:
                    results.append(x)
        
        return results























