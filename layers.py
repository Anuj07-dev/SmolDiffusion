import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
  def __init__(self, in_channels:int, out_channels:int, is_res:bool = False) -> None:
    super().__init__()

    self.same_channels = in_channels == out_channels    #True for residual connection
    self.is_res = is_res                                #Flag for whether to use residual connection

    self.conv1 = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 3, 1, 1), #3x3 kernel with stride 1 and padding 1
          nn.BatchNorm2d(out_channels),
          nn.GELU(),
    )

    self.conv2 = nn.Sequential(
          nn.Conv2d(out_channels, out_channels, 3, 1, 1), #3x3 kernel with stride 1 and padding 1
          nn.BatchNorm2d(out_channels),
          nn.GELU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    if self.is_res:                                     #If using residual connection
      x1 = self.conv1(x)
      x2 = self.conv2(x1)

      if self.same_channels:
        out = x + x2

      else:                                             #1x1 convolution to match dimensions
        shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
        out = shortcut(x) + x2


      return out / 1.1414                                #Normalization output tensor

    else:
      x1 = self.conv1(x)
      x2 = self.conv2(x1)
      return x2

  def get_out_channels(self):                           #Method to get the number of output channels
    return self.conv2[0].out_channels

  def set_out_channels(self,out_channels):              #Method to set the number of output channels
    self.conv1[0].out_channels, self.conv2[0].in_channels, self.conv2[0].out_channels = out_channels, out_channels, out_channels



class UnetDown(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()


    #Each layer: 2 residual conv block layers and one MaxPool2d layer for downsampling
    layers = [
      ResidualConvBlock(in_channels, out_channels), 
      ResidualConvBlock(out_channels, out_channels), 
      nn.MaxPool2d(2)]

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)


class UnetUp(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        ResidualConvBlock(out_channels, out_channels),
        ResidualConvBlock(out_channels, out_channels)
    ]

    self.model = nn.Sequential(*layers)


  def forward(self, x, skip):
    x = torch.cat((x, skip), 1)
    return self.model(x)


class EmbedFC(nn.Module):
  def __init__(self, input_dim, emb_dim):
    super().__init__()
    '''
    Generic one layer feed-forward network
    '''
    self.input_dim = input_dim
    layers = [
        nn.Linear(input_dim, emb_dim),
        nn.GELU(),
        nn.Linear(emb_dim,emb_dim)

    ]

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x.view(-1, self.input_dim))