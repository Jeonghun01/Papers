"""
    Load Data
      pre processing data
      post processing data

    Load Model

    Load loss

    Setting hyperparameters, gen, optimizer
    We define closure function to use L

    Train loop
      loss print
      image gen output save
"""

import torch
import torch.optim as optim

import numpy as np
from PIL import Image 

from Dataset import pre_processing, post_processing
from Model import StyleTransfer
from Loss import ContentLoss, StyleLoss

import os
from tqdm import tqdm


def train_model():
    
    content_image   = Image.open('./content.jpg')
    content_image   = pre_processing(content_image)
    style_image     = Image.open('./style.jpg')
    style_image     = pre_processing(style_image)

    style_transfer  = StyleTransfer()

    content_loss    = ContentLoss()
    style_loss      = StyleLoss()

    alpha           = 1
    beta            = 1e6
    learning_rate   = 1
    #gen = torch.randn(1, 3, 512, 512).requires_grad_(True)

    save_root = f'{alpha}_{beta}_{learning_rate}'
    os.makedirs(save_root, exist_ok = True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    style_transfer = style_transfer.to(device)
    content_image  = content_image.to(device)
    style_image    = style_image.to(device)
    gen = torch.randn(1, 3, 512, 512).to(device)
    gen.requires_grad_(True)
    

    optimizer = optim.LBFGS([gen], lr = learning_rate)

    def closure():
      optimizer.zero_grad()

      content_image_list = style_transfer.forward(content_image, 'Content')
      content_gen_list   = style_transfer.forward(gen, 'Content')
      style_image_list   = style_transfer.forward(style_image, 'Style')
      style_gen_list     = style_transfer.forward(gen, 'Style')

      loss_C             = 0
      loss_S             = 0
      total_loss         = 0

      for C, G in zip(content_image_list, content_gen_list):
         loss_C += content_loss.forward(C, G)
      for S, G in zip(style_image_list, style_gen_list):
         loss_S += style_loss.forward(S, G)
  
      total_loss = alpha * loss_C + beta * loss_S

      total_loss.backward()

      return total_loss
    

    epochs = 501
    for epoch in tqdm(range(epochs)):

      optimizer.step(closure)

      if epoch % 100 == 0:
        with torch.no_grad():
          # same as closure for computing loss
          content_image_list = style_transfer.forward(content_image, 'Content')
          content_gen_list   = style_transfer.forward(gen, 'Content')
          style_image_list   = style_transfer.forward(style_image, 'Style')
          style_gen_list     = style_transfer.forward(gen, 'Style')

          loss_C             = 0
          loss_S             = 0
          total_loss         = 0

          for C, G in zip(content_image_list, content_gen_list):
            loss_C += content_loss.forward(C, G)
          for S, G in zip(style_image_list, style_gen_list):
            loss_S += style_loss.forward(S, G)
  
          total_loss = alpha * loss_C + beta * loss_S

          print("Content Loss :", loss_C.item())
          print("Style Loss :", loss_S.item())
          print("Total Loss :", total_loss.item())

          gen_image:Image.Image = post_processing(gen)
          gen_image.save(os.path.join(save_root, f'{epoch}.jpg'))

if __name__ == "__main__":
   train_model()





    


