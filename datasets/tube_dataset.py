



# import imports
import os
import random
import numpy as np
from numpy.core.numeric import indices
import json

import torch
import torch.utils.data as data
from torch.utils.data import dataset
import torchvision.transforms as transforms

from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG

from transformations.dynamic_image_transformation import DynamicImage
# from transformations.temporal_transforms import CenterCrop, RandomCrop

from utils.global_var import *
from utils.utils import natural_sort
from utils.dataset_utils import imread, filter_data_without_tubelet, JSON_2_tube, check_no_tubes
from utils.tube_utils import JSON_2_videoDetections, JSON_2_tube, tube_2_JSON
from datasets.create_tube_sampler import get_sampler

from tubes.run_tube_gen import extract_tubes_from_video

class TubeDataset(data.Dataset):
    def __init__(self, cfg, make_fn, inputs_config, dataset, train_set):
        """Init a TubeDataset

        Args:
            cfg (Yaml): cfg.TUBE_DATASET
            make_fn ([type]): [description]
            inputs_config ([CnnInputConfig]): [description]
            dataset ([type]): [description]
            train_set ([bool]): Indicates if train set or val set
        Returns:
            [type]: [description]
        """
        self.cfg = cfg
        self.make_function = make_fn
        self.config = inputs_config
        self.dataset = dataset
        self.train_set = train_set
        if self.dataset == 'UCFCrime':
            self.paths, self.labels, _, self.annotations, self.num_frames = self.make_function()
        elif self.dataset == UCFCrimeReduced_DATASET:
            self.paths, self.labels, self.annotations, self.num_frames = self.make_function()
        else:
            self.paths, self.labels, self.annotations = self.make_function()
            self.paths, self.labels, self.annotations = filter_data_without_tubelet(self.paths, self.labels, self.annotations)

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = get_sampler(self.cfg, self.train_set)
        # if self.cfg.USE_TUBES:
        #     self.sampler = TubeCrop(tube_len=cfg.NUM_FRAMES,
        #                             central_frame=True,
        #                             max_num_tubes=cfg.NUM_TUBES,
        #                             # input_type=self.config['input_1'].itype,
        #                             sample_strategy=cfg.FRAMES_STRATEGY,
        #                             random=cfg.RANDOM,
        #                             box_as_tensor=False)
        # else:
        #     if self.train_set:
        #         self.sampler = RandomCrop(size=self.cfg.NUM_FRAMES,
        #                                   stride=1,
        #                                   input_type='rgb')
        #     else:
        #         self.sampler = CenterCrop(size=self.cfg.NUM_FRAMES,
        #                                   stride=1,
        #                                   input_type='rgb')
        if self.config['input_2'].itype == DYN_IMAGE:
            self.dynamic_image_fn = DynamicImage()
    
    def build_frame_name(self, path, frame_number, frames_names_list):
        if self.dataset == RWF_DATASET:
            return os.path.join(path,'frame{}.jpg'.format(frame_number+1))
        elif self.dataset == HOCKEY_DATASET:
            return os.path.join(path,'frame{:03}.jpg'.format(frame_number+1))
        elif self.dataset == RLVSD_DATASET:
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
        elif self.dataset == UCFCrime_DATASET:
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
        elif self.dataset == UCFCrimeReduced_DATASET:
            frame_idx = frame_number
            pth = os.path.join(path, frames_names_list[frame_idx])
            return pth
    
    def __format_bbox__(self, bbox):
        """
        Format a tube bbox: [x1,y1,x2,y2] to a correct format
        """
        [width, height] = self.cfg.SHAPE
        bbox = bbox[0:4]
        bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
        bbox = bbox.reshape(1,-1).astype(float)
        if self.cfg.BOX_AS_TENSOR:
            bbox = torch.from_numpy(bbox).float()
        return bbox
    
    def __scale_bbox__(self, bbox, img_size):
        """
        Format a tube bbox: [x1,y1,x2,y2] to a correct format
        """
        
        [width, height] = self.cfg.SHAPE
        w, h = img_size
        scale_w = w/width
        scale_h = h/height

        bbox = bbox[0:4]
        # bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
        bbox = np.array([bbox[0]/scale_w, bbox[1]/scale_h, bbox[2]/scale_w, bbox[3]/scale_h])
        
        bbox = bbox.reshape(1,-1).astype(float)
        if self.cfg.BOX_AS_TENSOR:
            bbox = torch.from_numpy(bbox).float()
        return bbox
    
    def load_input_1(self, path, frames_indices, frames_names_list):
        tube_images = []
        raw_clip_images = []
        images_t = None
        if self.config['input_1'].itype=='rgb':
            frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames_indices]
            # for j, fp in enumerate(frames_paths):
            #     print(j, ' ', fp)
            for i in frames_paths:
                img = imread(i)
                # print('img shape: ', img.size)
                tube_images.append(img)
            
            raw_clip_images = tube_images.copy()
            if self.config['input_1'].spatial_transform:
                fake_boxes = [np.array([10,10,100,100]).reshape(1,-1).astype(float) for i in range(len(tube_images))]
                images_t, tube_boxes_t, t_combination = self.config['input_1'].spatial_transform(tube_images, fake_boxes)
       
        return images_t, tube_boxes_t, raw_clip_images, t_combination
    
    def load_input_1_from_tube(self, path, frames_indices, frames_names_list, sampled_tube):
        tube_images = []
        raw_clip_images = []
        tube_images_t = None
        tube_boxes = []
        if self.config['input_1'].itype=='rgb':
            frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames_indices]
            # for j, fp in enumerate(frames_paths):
            #     print(j, ' ', fp)
            for i in frames_paths:
                img = imread(i)
                # print('img shape: ', img.size)
                tube_images.append(img)
                _, frame_name = os.path.split(i)
                
                try:
                    box_idx = sampled_tube['frames_name'].index(frame_name)
                except Exception as e:
                    print("\nOops!", e.__class__, "occurred.")
                    print("sampled_tube['frames_name']: {}, frame: {} , sampled_indices: {}, path: {}".format(sampled_tube['frames_name'], frame_name, frames_indices, path))
                    exit()
                tube_boxes.append(box_idx)
            
            tube_boxes_raw_size = [sampled_tube['boxes'][b] for b in tube_boxes]
            # tube_boxes = [self.__format_bbox__(t) for t in tube_boxes_raw_size]
            tube_boxes = [np.array(t[0:4]).reshape(1,-1).astype(float) for t in tube_boxes_raw_size]
            tube_boxes = [np.where(t<0, 0, t).reshape(1,-1).astype(float) for t in tube_boxes]
            
            # print('\tube_boxes: ', tube_boxes, len(tube_boxes))
            # print('\t tube_images: ', type(tube_images), type(tube_images[0]))
            raw_clip_images = tube_images.copy()
            if self.config['input_1'].spatial_transform:
                tube_images_t, tube_boxes_t, t_combination = self.config['input_1'].spatial_transform(tube_images, tube_boxes.copy())
                for i,tb in enumerate(tube_boxes_t):
                    if tb.size(0)==0:
                        print('error in Resize: ', tb, t_combination)
                        img = np.array(tube_images[i])
                        box = tube_boxes[i]
                        w,h = img.shape[1], img.shape[0]
                        print('w,h :', w, h)
                        # img = letterbox_image(img, self.inp_dim)
                        inp_dim = 224
                        scale = min(inp_dim/h, inp_dim/w)
                        print('scale :', scale)
                        box[:,:4] *= (scale)
                        print('box :', box)
                        new_w = scale*w
                        new_h = scale*h
                    
                        del_h = (inp_dim - new_h)/2
                        del_w = (inp_dim - new_w)/2
                    
                        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
                        print('add_matrix :', add_matrix)
                        box[:,:4] += add_matrix
                        print('box :', box)
                        box = torch.from_numpy(box).float()
                        print('box tensor:', box, box.size())
                # print('Applied transforms: ', t_combination)
       
        return tube_images_t, tube_boxes_t, tube_boxes, tube_boxes_raw_size, raw_clip_images, t_combination

    def extract_tubes(self, 
                      video_path, 
                      pers_annotation, 
                      sequence, 
                      sequence_video_names,
                      label):
        # print('video_path: ', video_path)
        video_name = video_path.split('/')[-1]
        frame_s = sequence_video_names[0].split('_')[-1][0:-4]
        frame_e = sequence_video_names[-1].split('_')[-1][0:-4]
        
        tube_path = os.path.join(self.tube_folder, "{}_from_{}_to_{}_{}.json".format(video_name, frame_s, frame_e, label))
        # print('start_frame: {}, end_frame: {}/ tube_p:{}--label: {}'.format(sequence_video_names[0], sequence_video_names[-1], tube_path, label))
        if not os.path.isfile(tube_path):
            # print("tube not found, extracting at: ", tube_path)
            person_detections = JSON_2_videoDetections(pers_annotation)
            TUBE_BUILD_CONFIG['person_detections'] = person_detections
            if self.train_set:
                TUBE_BUILD_CONFIG['train_mode'] = True
            tubes, time = extract_tubes_from_video(sequence, sequence_video_names, MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG, None)
            tube_2_JSON(tube_path, tubes)
        else:
            # print('reusing tube: ', tube_path)
            tubes = JSON_2_tube(tube_path)
        return tubes

    def load_tube_from_file(self, annotation):
        if self.dataset == 'UCFCrime':
            return annotation
        else:
            if isinstance(annotation, list):
                video_tubes = annotation
            else:
                video_tubes = JSON_2_tube(annotation)
            assert len(video_tubes) >= 1, "No tubes in video!!!==>{}".format(annotation)
            return video_tubes

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        frames_names_list = os.listdir(path)
        frames_names_list = natural_sort(frames_names_list)
        # print('frames_names_list: ', frames_names_list)
        
        label = self.labels[index]
        annotation = self.annotations[index]
        video_images = []
        video_images_raw = []
        final_tube_boxes = []
        tube_boxes_raw = []
        if not self.cfg.USE_TUBES:
            indices_frames = list(range(len(frames_names_list)))
            frames_indices = self.sampler(indices_frames)
            images_t, fake_boxes_t, raw_clip_images, t_combination = self.load_input_1(path, frames_indices, frames_names_list)
            video_images = [torch.stack(images_t, dim=0)] #video_images[0]:  torch.Size([16, 224, 224, 3])
            # print('\t video_images[0]: ', video_images[0].size())
            final_tube_boxes = [fake_boxes_t[0]] #final_tube_boxes[0]:  torch.Size([1, 4])
            # print('\t final_tube_boxes[0]: ', final_tube_boxes[0].size())
        else:
            tubes_ = self.load_tube_from_file(annotation)
            # tubes_ = self.extract_tubes(path, pers_annotation, sampled_clip_indices, clip_frames, label)
            sampled_frames_indices, chosed_tubes = self.sampler(tubes_)

            # for i in range(len(sampled_frames_indices)):
            #     print('\ntube[{}] \n (1)frames_names_list: {}, \n(2)tube frames_name: {}, \n(3)sampled_frames_indices: {}'.format(i,frames_names_list, chosed_tubes[i]['frames_name'], sampled_frames_indices[i]))
            # print('sampled_frames_indices: ', sampled_frames_indices)
            # print('boxes_from_sampler: ', boxes, boxes[0].shape)
            
            for frames_indices, sampled_tube in zip(sampled_frames_indices, chosed_tubes):
                # print('\nload_input_1 args: ', path, frames_indices, boxes)
                tube_images_t, tube_boxes_t, tube_boxes, tube_boxes_raw_size, tube_raw_clip_images, t_combination = self.load_input_1_from_tube(path, frames_indices, frames_names_list, sampled_tube)
                video_images.append(torch.stack(tube_images_t, dim=0)) #added tensor: torch.Size([16, 224, 224, 3])
                video_images_raw.append(tube_raw_clip_images) #added PIL image
                # print('video_images[-1]: ', video_images[-1].size())
                #Box extracted from tube
                tube_box = None
                if self.cfg.BOX_STRATEGY == MIDDLE_BOX:
                    m = int(len(tube_boxes)/2) #middle box from tube
                    ##setting id to box
                    tube_box = tube_boxes_t[m]
                    id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
                    # print('\n', ' id_tensor: ', id_tensor,id_tensor.size())
                    # print(' tube_box_: ', tube_box, tube_box.size(), ' index: ', m)
                    # print(' tube_box: ', tube_boxes_raw_size[m], tube_box.size())
                    # print(' sampled_tube: ', sampled_tube)
                    if tube_box.size(0)==0:
                        print(' Here error: ', path, index, '\n',
                                tube_box, '\n', 
                                sampled_tube, '\n', 
                                frames_indices, '\n', 
                                tube_boxes_t, len(tube_boxes_t), '\n', 
                                tube_boxes, len(tube_boxes), '\n',
                                t_combination)
                        exit()
                    f_box = torch.cat([id_tensor , tube_box], dim=1).float()
                elif self.cfg.BOX_STRATEGY == UNION_BOX:
                    # all_boxes = [torch.from_numpy(t) for i, t in enumerate(tube_boxes_t)]
                    all_boxes = [t for i, t in enumerate(tube_boxes_t)]
                    # print('\nall boxes: ', all_boxes, len(all_boxes))
                    try:
                        all_boxes = torch.stack(all_boxes, dim=0).squeeze()
                    except RuntimeError as e:
                        print(str(e))
                        print('video: ', path)
                        for i in range(len(all_boxes)):
                            print('-b:',i , all_boxes[i], tube_boxes_t[i], tube_boxes_raw_size[i])
                        print('tube: ', sampled_tube)
                    mins, _ = torch.min(all_boxes, dim=0)
                    x1 = mins[0].unsqueeze(dim=0).float()
                    y1 = mins[1].unsqueeze(dim=0).float()
                    maxs, _ = torch.max(all_boxes, dim=0)
                    x2 = maxs[2].unsqueeze(dim=0).float()
                    y2 = maxs[3].unsqueeze(dim=0).float()
                    id_tensor = torch.tensor([0]).float()
                    
                    f_box = torch.cat([id_tensor , x1, y1, x2, y2]).float()
                elif self.cfg.BOX_STRATEGY == ALL_BOX:
                    f_box = [torch.cat([torch.tensor([i]).unsqueeze(dim=0), torch.from_numpy(t)], dim=1).float() for i, t in enumerate(tube_boxes)]
                    f_box = torch.stack(f_box, dim=0)
                final_tube_boxes.append(f_box)
        
        #load keyframes
        key_frames = []
        key_frames_raw = []
        if self.config['input_2'] is not None:
            for k in range(len(video_images)):
                if self.cfg.KEYFRAME_STRATEGY == DYNAMIC_IMAGE_KEYFRAME:
                    # key_frame, _ = self.load_input_2_di(sampled_frames_indices[k], path, frames_names_list)
                    key_frame = self.dynamic_image_fn(video_images[k])
                    # key_frames_raw.append(key_frame)
                    if self.cfg.KEYFRAME_CROP:
                        rect = final_tube_boxes[k][1:5]
                        key_frame = key_frame.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                        key_frame = key_frame.resize((self.cfg.SHAPE[0], self.cfg.SHAPE[1]))
                    # key_frames_raw.append(key_frame)
                    if self.config['input_2'].spatial_transform:
                        key_frame = self.config['input_2'].spatial_transform(key_frame)
                    key_frames_raw.append(transforms.ToPILImage()(key_frame))
                elif self.cfg.KEYFRAME_STRATEGY == RGB_MIDDLE_KEYFRAME:
                    m = int(video_images[k].size(0)/2) #using frames loaded from 3d branch
                    key_frame = video_images[k][m] #tensor torch.Size([224, 224, 3])
                    #TODO multiply per 255????
                    key_frame_pil = transforms.ToPILImage()(key_frame.permute(2,0,1)) #(224, 224)
                    # key_frames_raw.append(key_frame_pil)
                    if self.cfg.KEYFRAME_CROP:
                        rect = final_tube_boxes[k][1:5]
                        key_frame = key_frame_pil.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                        key_frame = key_frame.resize((self.cfg.SHAPE[0], self.cfg.SHAPE[1]))
                    else:
                        key_frame = key_frame_pil
                    
                    if self.config['input_2'].spatial_transform:
                        key_frame = self.config['input_2'].spatial_transform(key_frame) #torch.Size([3, 224, 224])
                        # print('key_frame transformed: ', key_frame.size())
                    
                    key_frames_raw.append(transforms.ToPILImage()(key_frame))
                else:
                    #TODO
                    print('Not implemented yet...')
                    exit()
                key_frames.append(key_frame)
        
        #padding
        if len(video_images)<self.cfg.NUM_TUBES:
            for i in range(self.cfg.NUM_TUBES-len(video_images)):
                video_images.append(video_images[len(video_images)-1])
                p_box = tube_boxes[len(tube_boxes)-1]
                tube_boxes.append(p_box)
                if self.config['input_2'] is not None:
                    key_frames.append(key_frames[-1])

        final_tube_boxes = torch.stack(final_tube_boxes, dim=0).squeeze()
        
        if len(final_tube_boxes.shape)==1:
            final_tube_boxes = torch.unsqueeze(final_tube_boxes, dim=0)
            # print('boxes unsqueeze: ', boxes)
        
        video_images = torch.stack(video_images, dim=0).permute(0,4,1,2,3)#.permute(0,2,1,3,4)
        key_frames = torch.stack(key_frames, dim=0)
        if not self.cfg.USE_TUBES:
            # video_images = video_images.squeeze()
            # key_frames = key_frames.squeeze()
            final_tube_boxes = final_tube_boxes.squeeze(dim=1)
        return final_tube_boxes, video_images, label, path, key_frames#, key_frames_raw

        # if self.config['input_2'] is not None:
        #     key_frames = torch.stack(key_frames, dim=0)
        #     if torch.isnan(key_frames).any().item():
        #         print('Detected Nan at: ', path)
        #     if torch.isinf(key_frames).any().item():
        #         print('Detected Inf at: ', path)
        #     return final_tube_boxes, video_images, label, path, key_frames#, key_frames_raw
        # else:
        #     return final_tube_boxes, video_images, label, path


