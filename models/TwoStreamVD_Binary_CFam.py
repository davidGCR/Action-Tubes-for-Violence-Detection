
from models._3d_backbones import *
from models._2d_backbones import Backbone2DResNet, MyResNet
from models.roi_extractor_3d import SingleRoIExtractor3D
from models.cfam import CFAMBlock 
from models.identity import Identity

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.roi_align import RoIAlign

class RoiPoolLayer(nn.Module):
    def __init__(self,roi_layer_type='RoIAlign',
                      roi_layer_output=8,
                      roi_with_temporal_pool=True,
                      roi_spatial_scale=16,
                      with_spatial_pool=True): #832):
        
        super(RoiPoolLayer, self).__init__()
        self.roi_op = SingleRoIExtractor3D(roi_layer_type=roi_layer_type,
                                            featmap_stride=roi_spatial_scale,
                                            output_size=roi_layer_output,
                                            with_temporal_pool=roi_with_temporal_pool)

        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.with_spatial_pool = with_spatial_pool
        
    
    def forward(self, x, bbox):
        #x: b,c,t,w,h
        batch, c, t, h, w = x.size()
        # print('before roipool: ', x.size(), ', bbox: ', bbox.size())
        x, _ = self.roi_op(x, bbox)
        # print('RoiPoolLayer after roipool: ', x.size())
        x = self.temporal_pool(x)
        # print('after temporal_pool: ', x.size())

        if self.with_spatial_pool:
            x = self.spatial_pool(x) #torch.Size([16, 528, 1, 1, 1]
            # print('after spatial_pool: ', x.size())   
            x = x.view(x.size(0),-1)
        return x

class TwoStreamVD_Binary_CFam_Eval(nn.Module):
  def __init__(self, model):
    super(TwoStreamVD_Binary_CFam_Eval, self).__init__()
    self.model = model
    self.model.avg_pool_2d = Identity()
    self.model.classifier = Identity()
    
    self.classifier = nn.Conv2d(512, 1, kernel_size=1, bias=False)
    self.avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
    # self.backbone.blocks[6] = Identity()
  
  def forward(self, x1, x2, bbox=None, num_tubes=0):
    x = self.model(x1, x2, bbox, num_tubes) #torch.Size([512])
    print('class backbone: ', x.size())
    x = self.classifier(x)
    x = self.avg_pool_2d(x)
    print('after avg2D: ', x.size())
    x = torch.squeeze(x)
    x = torch.sigmoid(x)
    return x

class TwoStreamVD_Binary_CFam(nn.Module):
    def __init__(self, cfg):
        super(TwoStreamVD_Binary_CFam, self).__init__()
        self.cfg = cfg
        # self.with_roipool = self.cfg.WITH_ROIPOOL #config['with_roipool']
        self._3d_stream = self.build_3d_backbone()
        
        if self.cfg._2D_BRANCH.ACTIVATE:
            # self._2d_stream = Backbone2DResNet(
            #     self.cfg._2D_BRANCH.NAME ,#config['2d_backbone'],
            #     self.cfg._2D_BRANCH.FINAL_ENDPOINT,#config['base_out_layer'],
            #     num_trainable_layers=self.cfg._2D_BRANCH.NUM_TRAINABLE_LAYERS#config['num_trainable_layers']
            #     )
            self._2d_stream = MyResNet(last_layer=self.cfg._2D_BRANCH.FINAL_ENDPOINT)
        
        if self.cfg._3D_BRANCH.WITH_ROIPOOL:
            self.roi_pool_3d = RoiPoolLayer(
                roi_layer_type=self.cfg._ROI_LAYER.TYPE,#config['roi_layer_type'],
                roi_layer_output=self.cfg._ROI_LAYER.OUTPUT,#config['roi_layer_output'],
                roi_with_temporal_pool=self.cfg._ROI_LAYER.WITH_TEMPORAL_POOL,#config['roi_with_temporal_pool'],
                roi_spatial_scale=self.cfg._ROI_LAYER.SPATIAL_SCALE,#config['roi_spatial_scale'],
                with_spatial_pool=self.cfg._ROI_LAYER.WITH_SPATIAL_POOL#False
                )
        else:
            # self.avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
        if self.cfg._2D_BRANCH.WITH_ROIPOOL:
            self.roi_pool_2d = RoIAlign(output_size=self.cfg._ROI_LAYER.OUTPUT,#config['roi_layer_output'],
                                        spatial_scale=self.cfg._ROI_LAYER.SPATIAL_SCALE,#config['roi_spatial_scale'],
                                        sampling_ratio=0,
                                        aligned=True
                                        )

        in_channels = self.cfg._CFAM_BLOCK.IN_CHANNELS
        out_channels = self.cfg._CFAM_BLOCK.OUT_CHANNELS if self.cfg._2D_BRANCH.ACTIVATE else self.cfg._HEAD.INPUT_DIM
        
        if self.cfg._CFAM_BLOCK.ACTIVATE:
            self.CFAMBlock = CFAMBlock(in_channels, out_channels)

        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
        if self.cfg._HEAD.NAME == 'binary':
            self.classifier = nn.Conv2d(out_channels, 2, kernel_size=1, bias=False)
        elif self.cfg._HEAD.NAME == 'regression':
            self.classifier = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)
    
    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
    
    def build_3d_backbone(self):
        if self.cfg._3D_BRANCH.NAME == 'i3d':
            backbone = BackboneI3D(
                self.cfg._3D_BRANCH.FINAL_ENDPOINT,
                self.cfg._3D_BRANCH.PRETRAINED_MODEL,
                freeze=self.cfg._3D_BRANCH.FREEZE_3D
                )
        elif self.cfg._3D_BRANCH.NAME == 'i3dv2':
            backbone = BackboneI3D_V2(
                self.cfg._3D_BRANCH.PRETRAINED_MODEL,
                freeze=self.cfg._3D_BRANCH.FREEZE_3D
                )

            # print('i3dv2 model:')
            # print(backbone)
        elif self.cfg._3D_BRANCH.NAME == 'x3d':
            backbone = BackboneX3D(
                self.cfg._3D_BRANCH.PRETRAINED_MODEL,
                freeze=self.cfg._3D_BRANCH.FREEZE_3D
                )
        elif self.cfg._3D_BRANCH.NAME == '3dresnet':
            backbone = Backbone3DResNet(
                pretrained=self.cfg._3D_BRANCH.PRETRAINED_MODEL,
                freeze=self.cfg._3D_BRANCH.FREEZE_3D
            )
        return backbone
    
    def forward_3d_branch(self, x1, bbox=None, num_tubes=0):
        batch, c, t, h, w = x1.size()
        x_3d = self._3d_stream(x1)
        # print('output_3dbackbone: ', x_3d.size())
        if self.cfg._3D_BRANCH.WITH_ROIPOOL:
            batch = int(batch/num_tubes)
            x_3d = self.roi_pool_3d(x_3d,bbox)#torch.Size([8, 528])
            # print('3d after roipool: ', x_3d.size())
            x_3d = torch.squeeze(x_3d, dim=2)
            
            b_1, c_1, w_1, h_1 = x_3d.size()

            if self.cfg._HEAD.NAME == 'binary':
                x_3d = x_3d.view(batch, num_tubes, c_1, w_1, h_1)
                x_3d = x_3d.max(dim=1).values
                # print('after tmp max pool: ', x_3d.size())
                x_3d = self.classifier(x_3d)
                # print('after classifier conv: ', x_3d.size())
                x_3d = self.avg_pool_2d(x_3d)
                # print('after avg2D: ', x_3d.size())
                x_3d = torch.squeeze(x_3d)
        return x_3d
        
    def forward(self, x1, x2, bbox=None, num_tubes=0):
        if not self.cfg._2D_BRANCH.ACTIVATE:
            return self.forward_3d_branch(x1, bbox, num_tubes)
        batch, c, t, h, w = x1.size()
        x_3d = self._3d_stream(x1) #torch.Size([2, 528, 4, 14, 14])
        x_2d = self._2d_stream(x2) #torch.Size([2, 1024, 14, 14])

        # print('output_3dbackbone: ', x_3d.size())
        # print('output_2dbackbone: ', x_2d.size())
        if self.cfg._3D_BRANCH.WITH_ROIPOOL:
            batch = int(batch/num_tubes)
            x_3d = self.roi_pool_3d(x_3d,bbox)#torch.Size([8, 528])
            # print('3d after roipool: ', x_3d.size())
            x_3d = torch.squeeze(x_3d, dim=2)
            # x_3d = torch.squeeze(x_3d)
            
        else:
            x_3d = self.temporal_pool(x_3d)
            # print('3d after tmppool: ', x_3d.size())
            x_3d = torch.squeeze(x_3d)
            # print('3d after squeeze: ', x_3d.size())
        
        if self.cfg._2D_BRANCH.WITH_ROIPOOL:
            x_2d = self.roi_pool_2d(x_2d, bbox)
            # print('2d after roipool: ', x_2d.size())

        x = torch.cat((x_3d,x_2d), dim=1) #torch.Size([8, 1552, 8, 8])
        # print('after cat 2 branches: ', x.size())
        x = self.CFAMBlock(x) #torch.Size([8, 145, 8, 8])
        # print('after CFAMBlock: ', x.size())

        if self.cfg._3D_BRANCH.WITH_ROIPOOL:
            b_1, c_1, w_1, h_1 = x.size()

            if self.cfg._HEAD.NAME == 'binary':
                x = x.view(batch, num_tubes, c_1, w_1, h_1)
                x = x.max(dim=1).values
                # print('after tmp max pool: ', x.size())
                x = self.classifier(x)
                # print('after classifier conv: ', x.size())
                x = self.avg_pool_2d(x)
                # print('after avg2D: ', x.size())
                x = torch.squeeze(x)
            elif self.cfg._HEAD.NAME == 'regression':
                x = self.classifier(x)
                # print('after classifier conv: ', x.size())
                x = self.avg_pool_2d(x)
                # print('after avg2D: ', x.size())
                x = torch.squeeze(x)
                x = torch.sigmoid(x)
                # print('after sigmoid: ', x.size(), x)
                x = x.view(batch, num_tubes, -1)
                x = x.max(dim=1).values
                x = torch.squeeze(x)
                # print('after max: ', x.size(), x)
        else:
            batch = int(batch/num_tubes)
            b_1, c_1, w_1, h_1 = x.size()
            # print('before tube max pool: ',b_1, c_1, w_1, h_1)
            x = x.view(batch, num_tubes, c_1, w_1, h_1)
            x = x.max(dim=1).values
            x = self.classifier(x)
            # print('after classifier: ', x.size())
            x = self.avg_pool_2d(x)
            # print('after avg_pool_2d: ', x.size())
            x = torch.squeeze(x)
            # print('after squeeze: ', x.size())
        return x
        

if __name__=='__main__':

    

    
    
    print('------- ViolenceDetector --------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TwoStreamVD_Binary_CFam(config=None).to(device)
    # # model = ViolenceDetectorRegression(aggregate=True).to(device)
    batch = 2
    tubes = 4
    input_1 = torch.rand(batch*tubes,3,8,224,224).to(device)
    input_2 = torch.rand(batch*tubes,3,224,224).to(device)

    rois = torch.rand(batch*tubes, 5).to(device)
    rois[0] = torch.tensor([0,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)#torch.tensor([1, 14, 16, 66, 70]).to(device)
    rois[1] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[2] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[3] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[4] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[5] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[6] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[7] = torch.tensor([1, 34, 14, 85, 77]).to(device)

    output = model(input_1, input_2, rois, tubes)
    # output = model(input_1, input_2, None, None)
    # output = model(input_1, rois, tubes)
    print('output: ', output, output.size())
    
    # regressor = ViolenceDetectorRegression().to(device)
    # input_1 = torch.rand(batch*tubes,3,16,224,224).to(device)
    # output = regressor(input_1, rois, tubes)
    # print('output: ', output.size())

    # model = ResNet2D_Stream(config=TWO_STREAM_CFAM_CONFIG).to(device)
    # batch = 2
    # tubes = 3
    # input_2 = torch.rand(batch*tubes,3,224,224).to(device)
    # rois = torch.rand(batch*tubes, 5).to(device)
    # rois[0] = torch.tensor([0, 34, 14, 85, 77]).to(device)
    # rois[1] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[0] = torch.tensor([2, 34, 14, 85, 77]).to(device)
    # rois[1] = torch.tensor([3, 34, 14, 85, 77]).to(device)
    # rois[0] = torch.tensor([4, 34, 14, 85, 77]).to(device)
    # rois[1] = torch.tensor([5, 34, 14, 85, 77]).to(device)

    # output = model(input_2, rois, tubes)
    # print('output: ', output.size())