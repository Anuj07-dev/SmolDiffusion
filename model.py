import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ResidualConvBlock, UnetDown, UnetUp, EmbedFC

class ContextUnet(nn.Module):
  def __init__(self, in_channels, n_feat=256, n_cfeat=10, height= 28):
    super().__init__()

    self.in_channels = in_channels  #no. of input channels
    self.n_feat = n_feat            #no. of intermediate feature maps
    self.n_cfeat = n_cfeat          #no. of context features
    self.h = height                 #(H, W) on input image, H == W and H%4 = 0

    # print(f"in_channels: {self.in_channels}, n_feat: {self.n_feat}, n_cfeat: {self.n_cfeat}, height: {self.h}")

    #Initialize the initial conv layer
    self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
    # print(f"Inital conv layer:")
    # print(f"Input: (in_channels, n_feat) -> {in_channels, n_feat}")
    # print(f"Output: init_conv {self.init_conv}")

    #Initialize the down-sampling path of the U-Net
    self.down1 = UnetDown(n_feat, n_feat)     #[10, 256, 8, 8]
    self.down2 = UnetDown(n_feat, 2*n_feat)   #[10, 256, 4, 4]                                      #Check dimentions
    # print(f"Down1 layer:")
    # print(f"Input: (n_feat, n_feat) -> {n_feat, n_feat}")
    # print(f"Output: down1 {self.down1}")


    # print(f"Down2 layer:")
    # print(f"Input: (n_feat, 2*n_feat) -> {n_feat, 2*n_feat}")
    # print(f"Output: down2 {self.down2}")

    self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())                                         #Try 7 instead of 4

    # print(f"to_vec layer:")
    # print(f"Output: {self.to_vec}")

    #Embed the timestamp and context labels with a one-layer fully connected neural network
    self.timeembed1 = EmbedFC(1, 2*n_feat)
    self.timeembed2 = EmbedFC(1, 1*n_feat)
    self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
    self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

    # print(f"Timestamp embed 1 layer: {self.timeembed1}")
    # print(f"Timestamp embed 2 layer: {self.timeembed2}")

    # print(f"Context embed 1 layer: {self.contextembed1}")
    # print(f"Context embed 2 layer: {self.contextembed2}")

    #Initialize the up-sampling path of the U-Net with three levels
    self.up0 = nn.Sequential(
        nn.ConvTranspose2d(2*n_feat, 2*n_feat, self.h//4, self.h//4),
        nn.GroupNorm(8, 2 * n_feat),
        nn.ReLU(),
    )
    # print(f"Up0 layer:")
    # print(f"Input: (2*n_feat) -> {2*n_feat}")
    # print(f"Output: (2*n_feat) -> {2*n_feat}, kernel_size -> {self.h//4}, stride -> {self.h//4}")
    # print(f"Output: up0 {self.up0}")

    self.up1 = UnetUp(4*n_feat, n_feat)
    self.up2 = UnetUp(2*n_feat, n_feat)


    # print(f"Up1 layer:")
    # print(f"Input: (4*n_feat, n_feat) -> {4*n_feat, n_feat}")
    # print(f"Output: up1 {self.up1}")


    # print(f"Up2 layer:")
    # print(f"Input: (2*n_feat, n_feat) -> {2*n_feat, n_feat}")
    # print(f"Output: up2 {self.up2}")

    #Initialize the final convolutional layers to map the same number of channels as the input image
    self.out = nn.Sequential(
        nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), #reduces the number of feature maps
        nn.GroupNorm(8, n_feat),                #Normalize
        nn.ReLU(),
        nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), #map to the same number of channels as input

    )
    # print(f"Out layer")
    # print(f"Output: out {self.out}")

  def forward(self, x, t, c=None):
    '''
    x : (batch, n_feat, h, w) -> input image
    t : (batch, n_cfeat)      -> timestamp
    c : (batch, n_classes)    -> context label
    '''
    # print("------>")
    # print(x.shape, t.shape)
    x = self.init_conv(x)           #Initial conv layer
    down1 = self.down1(x)           #Down sampling 1 [10, 256, 8, 8]
    down2 = self.down2(down1)       #Down sampling 2 [10, 256, 4, 4]

    hiddenvec = self.to_vec(down2) #feature map to vector and apply an activation

    if c is None:
      c = torch.zeros(x.shape[0], self.n_cfeat)
    c = c.to(x)
    cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1) #Embed context1 (batch, 2*n_feat, 1, 1)
    temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)    #Embed timestamp1
    cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)     #Embed context2
    temb2 = self.timeembed2(t).view(-1,self.n_feat, 1, 1)         #Embed timestamp2


    up1 = self.up0(hiddenvec)
    up2 = self.up1(cemb1*up1 + temb1, down2)                      #Add and multiply embeddings
    up3 = self.up2(cemb2*up2 + temb2, down1)
    out = self.out(torch.cat((up3, x), 1))
    return out
