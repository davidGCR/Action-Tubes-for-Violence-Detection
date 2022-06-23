import add_libs
import os

import numpy as np
import torch
from configs.defaults import get_cfg_defaults
from configs.tube_config import MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG
from datasets.collate_fn import my_collate, my_collate_video
from datasets.dataloaders import two_stream_transforms
from models.TwoStreamVD_Binary_CFam import (TwoStreamVD_Binary_CFam,
                                            TwoStreamVD_Binary_CFam_Eval)

from torch.utils.data import DataLoader
from utils.global_var import *
from datasets.onevideo_dataset import VideoDemo
from utils.utils import get_torch_device, load_checkpoint
from tubes.run_tube_gen import extract_tubes_from_video
from utils.tube_utils import JSON_2_tube, JSON_2_videoDetections

def score_tubes_by_motion(h_path):
    from datasets.make_dataset_handler import load_make_dataset
    from models.resnet import ResNet
    from utils.tube_utils import tube_2_JSON
    
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(WORK_DIR / "configs/DI_MODEL.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    cfg.DATA.DATASET = 'RWF-2000'
    cfg.DATA.CV_SPLIT = 1
    cfg.DATA.LOAD_GROUND_TRUTH = False
    
    cfg.TUBE_DATASET.NUM_FRAMES = 16
    # cfg.TUBE_DATASET.NUM_TUBES = 3
    cfg.TUBE_DATASET.RANDOM = False
    cfg.TUBE_DATASET.FRAMES_STRATEGY =   0
    cfg.TUBE_DATASET.BOX_STRATEGY = 1 #0:middle, 1: union, 2: all
    cfg.TUBE_DATASET.KEYFRAME_STRATEGY = 3 #0: rgb middle frame , 3:dynamic_images
    cfg.TUBE_DATASET.KEYFRAME_CROP = True
    cfg.MODEL.INFERENCE.CHECKPOINT_PATH = r'C:\Users\David\Desktop\DATASETS\Pretrained_Models\DI_MODEL_RESTNET_save_at_epoch-58.chk'
    
    make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=-1,
                                        train=True,
                                        category=2,
                                        shuffle=False)
    make_dataset_val = load_make_dataset(cfg.DATA,
                                    env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                    min_clip_len=-1,
                                    train=False,
                                    category=2,
                                    shuffle=False)
    
    # paths, labels, annotations = make_dataset_train()
    paths, labels, annotations = make_dataset_val()
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    _device = get_torch_device()
    model = ResNet().to(_device)
    model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    model.eval()
    for path, label, annotation in zip(paths, labels, annotations):
        print('\n', path, '\n', label, '\n', annotation)
        
        
        out_annotation_scored = Path(annotation)
        file = out_annotation_scored.name
        clase = out_annotation_scored.parents[0].name
        split = out_annotation_scored.parents[1].name
        dataset = out_annotation_scored.parents[2].name
        folder = out_annotation_scored.parents[4]/Path('ActionTubesV2Scored')
        out_annotation_scored = folder/dataset/split/clase/file
        
        # print('Out file: ', out_annotation_scored)
        
        vd = VideoDemo(cfg=cfg.TUBE_DATASET,
                        path=path,
                        tub_file=annotation,
                        tub_cfg=TUBE_BUILD_CONFIG,
                        mot_cgf=MOTION_SEGMENTATION_CONFIG,
                        ped_file=None,
                        vizualize_tubes=True,
                        vizualize_keyframe=False,
                        transformations=transforms_config_val
                        )
        scored_tubes = []
        for tube_data in vd:
            f_box, tube_images_t, key_frame, tube = tube_data
            key_frame = torch.unsqueeze(key_frame, dim=0).to(_device)
            # print('keyframe: ', key_frame.size())
            # print('\ntube score:', tube['score'])
            pred = model(key_frame)
            # print('pred: ', pred, pred.size())
            max_scores, predicted = torch.max(pred, 1)
            # print('max_scores: ', max_scores)
            # print('label: ', predicted)
            
            # probs = torch.sigmoid(pred)
            # print("probs: ", probs, probs.size())
            
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(pred)
            # print("probs softmax: ", probabilities, probabilities.size())
            
            score = probabilities.detach().cpu().numpy()[0,1]
            tube['score'] = str(score)
            # print('score: ', score, tube['score'])
            scored_tubes.append(tube)
        if not os.path.isdir(out_annotation_scored.parents[0]): #Create folder of split
            os.makedirs(out_annotation_scored.parents[0])
        
        tube_2_JSON(out_annotation_scored, scored_tubes)
   
def violence_localization(h_path):
    from datasets.make_dataset_handler import load_make_dataset
    # from models.resnet import ResNet
    from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam
    from utils.tube_utils import tube_2_JSON
    
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(WORK_DIR / "configs/DI_MODEL.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    cfg.DATA.DATASET = 'RWF-2000'
    cfg.DATA.CV_SPLIT = 1
    cfg.DATA.LOAD_GROUND_TRUTH = False
    
    cfg.TUBE_DATASET.NUM_FRAMES = 16
    # cfg.TUBE_DATASET.NUM_TUBES = 3
    cfg.TUBE_DATASET.RANDOM = False
    cfg.TUBE_DATASET.FRAMES_STRATEGY =   0
    cfg.TUBE_DATASET.BOX_STRATEGY = 1 #0:middle, 1: union, 2: all
    cfg.TUBE_DATASET.KEYFRAME_STRATEGY = 3 #0: rgb middle frame , 3:dynamic_images
    cfg.TUBE_DATASET.KEYFRAME_CROP = True
    cfg.MODEL.INFERENCE.CHECKPOINT_PATH = r'C:\Users\David\Desktop\DATASETS\Pretrained_Models\TWOSTREAM+I3Dv1+ResNet50+CFAM+TubesScored-RWF-save_at_epoch-35.chk'
    
    make_dataset_val = load_make_dataset(cfg.DATA,
                                    env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                    min_clip_len=-1,
                                    train=False,
                                    category=2,
                                    shuffle=False)
    
    # paths, labels, annotations = make_dataset_train()
    paths, labels, annotations = make_dataset_val()
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    _device = get_torch_device()
    model = TwoStreamVD_Binary_CFam(cfg).to(_device)
    model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    model.eval()
    for path, label, annotation in zip(paths, labels, annotations):
        print('\n', path, '\n', label, '\n', annotation)
        
        out_annotation_scored = Path(annotation)
        file = out_annotation_scored.name
        clase = out_annotation_scored.parents[0].name
        split = out_annotation_scored.parents[1].name
        dataset = out_annotation_scored.parents[2].name
        folder = out_annotation_scored.parents[4]/Path('ActionTubesV2Scored')
        out_annotation_scored = folder/dataset/split/clase/file
        
        # print('Out file: ', out_annotation_scored)
        
        vd = VideoDemo(cfg=cfg.TUBE_DATASET,
                        path=path,
                        tub_file=annotation,
                        tub_cfg=TUBE_BUILD_CONFIG,
                        mot_cgf=MOTION_SEGMENTATION_CONFIG,
                        ped_file=None,
                        vizualize_tubes=True,
                        vizualize_keyframe=False,
                        transformations=transforms_config_val
                        )
        scored_tubes = []
        for tube_data in vd:
            f_box, tube_images_t, key_frame, tube = tube_data
            key_frame = torch.unsqueeze(key_frame, dim=0).to(_device)
            # print('keyframe: ', key_frame.size())
            # print('\ntube score:', tube['score'])
            pred = model(key_frame)
            # print('pred: ', pred, pred.size())
            max_scores, predicted = torch.max(pred, 1)
            # print('max_scores: ', max_scores)
            # print('label: ', predicted)
            
            # probs = torch.sigmoid(pred)
            # print("probs: ", probs, probs.size())
            
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(pred)
            # print("probs softmax: ", probabilities, probabilities.size())
            
            score = probabilities.detach().cpu().numpy()[0,1]
            tube['score'] = str(score)
            # print('score: ', score, tube['score'])
            scored_tubes.append(tube)
        if not os.path.isdir(out_annotation_scored.parents[0]): #Create folder of split
            os.makedirs(out_annotation_scored.parents[0])
        
        tube_2_JSON(out_annotation_scored, scored_tubes)

    
def demo(args):
    h_path = HOME_WINDOWS
    cfg = get_cfg_defaults()
    cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3DRoiPool_2DRoiPool.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    cfg.MODEL.INFERENCE.CHECKPOINT_PATH = r'C:\Users\David\Desktop\DATASETS\Pretrained_Models\TWOSTREAM+I3Dv1+ResNet50+CFAM+TubesScored-RWF-save_at_epoch-35.chk'
    # _6-B11R9FJM_2 (TP)
    # 0_DzLlklZa0_5 (TN)
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    _device = get_torch_device()
    # model = TwoStreamVD_Binary_CFam(cfg.MODEL).to(_device)
    # model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    # model.eval()
        
    vd = VideoDemo(
        cfg=cfg.TUBE_DATASET,
        path=args.video_folder,
        # tub_file=r"C:\Users\David\Desktop\DATASETS\ActionTubesV2Scored\RWF-2000\{}\{}\{}.json".format(set_, category, video),
        tub_cfg=TUBE_BUILD_CONFIG,
        mot_cgf=MOTION_SEGMENTATION_CONFIG,
        ped_file=args.pd_file,
        vizualize_tubes=False,
        out_file=args.out_file,
        save_folder=args.save_folder,
        # vizualize_keyframe=True,
        transformations=transforms_config_val
    )
    
    if args.plot:
        vd.tub_file = args.out_file
        vd.plot_gen_tubes()
        

import argparse


        
if __name__=='__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--video_folder', type=str, required=True)
    parser.add_argument('--pd_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--plot', type=bool, default=False)
    # Parse the argument
    args = parser.parse_args()
    demo(args)
    
    # h_path = HOME_WINDOWS
    # score_tubes(h_path)
