import torch
from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam
from torchsummary import summary
from utils.utils import count_parameters
from ptflops import get_model_complexity_info

def see_models():
    # model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50')
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    params_num = count_parameters(model)
    print("Num parameters: ", params_num)
    macs, params = get_model_complexity_info(model, (3, 16, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    


def debug_model(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TwoStreamVD_Binary_CFam(cfg).to(device)

    params_num = count_parameters(model)
    print("Num parameters: ", params_num)
    # print(model)
    print('------- ViolenceDetector --------')
    
    # # model = ViolenceDetectorRegression(aggregate=True).to(device)
    batch = 1
    tubes = 3
    input_1 = torch.rand(batch*tubes,3,8,224,224).to(device)
    input_2 = torch.rand(batch*tubes,3,224,224).to(device)

    rois = torch.rand(batch*tubes, 5).to(device)
    rois[0] = torch.tensor([0,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)#torch.tensor([1, 14, 16, 66, 70]).to(device)
    rois[1] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[2] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[3] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[4] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[5] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[6] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[7] = torch.tensor([1, 34, 14, 85, 77]).to(device)

    output = model(input_1, input_2, rois, tubes)
    # output = model(input_1, input_2, None, None)
    # output = model(input_1, rois, tubes)
    print('output: ', output.size())

    # summary(model, [(1, 3, 3, 8, 224, 224), (3, 3, 224, 224)])
