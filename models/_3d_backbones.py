# import os
# import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/src/VioNet')
import torch
import torch.nn as nn
from models.i3d import InceptionI3d
from models.identity import Identity


class BackboneI3D_V2(nn.Module):
  def __init__(self, pretrained=None, freeze=False):
    super().__init__()
    if pretrained:
      # self.backbone = torch.hub.load('I3D_8x8_R50.pyth', 'i3d_r50', source='local')
      self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
    # else:
    #   self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    self.backbone.blocks[5] = Identity()
    self.backbone.blocks[6] = Identity()
  
  def forward(self, x):
    x = self.backbone(x)
    # x = self.roi_layer(x, bbox)
    return x

class BackboneX3D(nn.Module):
  def __init__(self, pretrained=None, freeze=False):
    super().__init__()
    if pretrained:
      # self.backbone = torch.hub.load('I3D_8x8_R50.pyth', 'i3d_r50', source='local')
      self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

    self.backbone.blocks[4] = Identity()
    self.backbone.blocks[5] = Identity()
  
  def forward(self, x):
    x = self.backbone(x)
    # x = self.roi_layer(x, bbox)
    return x

class Backbone3DResNet(nn.Module):
  def __init__(self, pretrained=None, freeze=False):
    super().__init__()
    if pretrained:
      self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    # else:
    #   self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    self.backbone.blocks[4] = Identity()
    self.backbone.blocks[5] = Identity()
    # self.roi_layer = RoiPoolLayer()
  
  def forward(self, x):
    x = self.backbone(x)
    # x = self.roi_layer(x, bbox)
    return x


class BackboneI3D(nn.Module):
  def __init__(self, final_endpoint, pretrained=None, freeze=False):
    super().__init__()
    self.backbone = InceptionI3d(2, in_channels=3, final_endpoint=final_endpoint)
    if pretrained:
      load_model_path = pretrained
      state_dict = torch.load(load_model_path)
      self.backbone.load_state_dict(state_dict,  strict=False)
    if freeze:
      print('Freezing 3d branch!!!')
      for param in self.backbone.parameters():
          param.requires_grad = False
  
  def forward(self, x):
    x = self.backbone(x)
    return x

if __name__=='__main__':
  # backbone = Backbone3DResNet()
  backbone = BackboneI3D(final_endpoint='Mixed_5b', pretrained=None)
  input = torch.rand(4,3,8,224,224)
  output=backbone(input)
  print('out: ', output.size())


