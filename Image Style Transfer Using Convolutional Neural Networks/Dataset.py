"""
    Pre - Image resize (512, 512)
          Image -> Tensor
          Normalize (VGG19)
          reshape (c, h, w) -> (b, c, h, w)

    Post - Tensor -> numpy
           reshape 
           gen : (b, c, h, w)
           -> (c, h, w) -> (h, w, c)
           Denormalization
           numpy -> Image
"""

import torch
from torchvision import transforms

import numpy as np
from PIL import Image 


# VGG19 Normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def pre_processing(image:Image.Image) -> torch.Tensor:
    preprocessing = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])
    image_tensor:torch.Tensor = preprocessing(image)
    
    return image_tensor.unsqueeze(0)


def post_processing(tensor:torch.Tensor) -> Image.Image:
    image:np.ndarray = tensor.to('cpu').detach().numpy()
    image = image.squeeze() # (c, h, w) because b = 1
    image = image.transpose(1, 2, 0) # (h, w, c)
    image = image * std + mean

    image = image.clip(0,1) * 255
    image = image.astype(np.uint8)

    return Image.fromarray(image) # numpy to PIL image


