import os
import re
import torch.utils.data as data
from operator import itemgetter
import torch
import numpy as np

from utils.dataset_utils import imread
from utils.tube_utils import JSON_2_videoDetections
from utils.utils import natural_sort
from utils.global_var import *

from datasets.tube_crop import TubeCrop
from transformations.dynamic_image_transformation import DynamicImage


class UCFCrime2LocalDataset(data.Dataset):
    """
    Load tubelets from one video
    Use to extract features tube-by-tube from just a video
    """

    def __init__(
        self, 
        root,
        path_annotations,
        abnormal,
        persons_detections_path,
        transform=None,
        clip_len=25,
        clip_temporal_stride=1):
        # self.dataset_root = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
        # self.split = 'anomaly',
        # self.video = 'Arrest036(2917-3426)',
        # self.p_d_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
        # self.gt_ann_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos'

        # self.dataset_root = dataset_root
        # self.split = split
        # self.video = video
        # self.p_d_path = p_d_path
        # self.gt_ann_path = gt_ann_path
        # self.transform = transform
        # self.person_detections = JSON_2_videoDetections(p_d_path)
        # self.tubes = extract_tubes_from_video(
        #     self.dataset_root,
        # )
        self.clip_temporal_stride = clip_temporal_stride
        self.clip_len = clip_len
        self.root = root
        self.path_annotations = path_annotations
        self.abnormal = abnormal
        self.make_dataset = MakeUCFCrime2LocalClips(root, path_annotations, abnormal)
        self.paths, self.labels, self.annotations = self.make_dataset()

        self.persons_detections_path = persons_detections_path

    def __len__(self):
        return len(self.paths)
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]

        return indices_segments

    def generate_tube_proposals(self, path, frames):
        tmp = path.split('/')
        split = tmp[-2]
        video = tmp[-1]
        p_d_path = os.path.join(self.persons_detections_path, split, video)
        person_detections = JSON_2_videoDetections(p_d_path)
        tubes = extract_tubes_from_video(
            self.root,
            person_detections,
            frames,
            # {'wait': 200}
            )
        return tubes

    def __getitem__(self, index):
        path = self.paths[index]
        ann = self.annotations[index]
        sp_annotations_gt = self.make_dataset.ground_truth_boxes(path, ann)

        video_clips = self.get_video_clips(path)
        return video_clips, path, ann, sp_annotations_gt

class UCFCrime2LocalVideoDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        path,
        sp_annotation,
        clip_len=25,
        clip_temporal_stride=1,
        tubes=None,
        transformations=None):
        self.cfg = cfg
        self.path = path
        self.sp_annotation = sp_annotation
        self.clip_len = clip_len
        self.clip_temporal_stride = clip_temporal_stride

        self.clips = self.get_video_clips(self.path)
        self.video_name = path.split('/')[-1]
        self.clase = path.split('/')[-2]
        self.tubes = tubes
        self.transformations = transformations
        #TODO
        self.label = 1
        self.sampler = TubeCrop(tube_len=cfg.NUM_FRAMES,
                                central_frame=True,
                                max_num_tubes=0, #all tubes
                                input_type=self.transformations['input_1'].itype,
                                sample_strategy=cfg.FRAMES_STRATEGY,
                                random=cfg.RANDOM,
                                box_as_tensor=False)
        if self.transformations['input_2'].itype == DYN_IMAGE:
            self.dynamic_image_fn = DynamicImage()
    
    def __len__(self):
        return len(self.clips)
    
    def split_list(self, lst, n):  
        for i in range(0, len(lst), n): 
            yield lst[i:i + n] 
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)

        # indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        # indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]
        # real_clip_len = self.clip_len*self.clip_temporal_stride
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = list(self.split_list(indices, self.clip_len)) 

        return indices_segments
    
    def load_sp_annotations(self, frames, ann_path):
        frames_numbers = [int(re.findall(r'\d+', f)[0]) for f in frames]
        frames_numbers.sort()
        annotations = []
        with open(ann_path) as fid:
            lines = fid.readlines()
            ss = 1 if lines[0].split()[5] == '0' else 0
            for line in lines:
                # v_name = line.split()[0]
                # print(line.split())
                ann = line.split()
                frame_number = int(ann[5]) + ss
                valid = ann[6]
                if valid == '0' and frame_number in frames_numbers:
                    annotations.append(
                        {
                            "frame": frame_number,
                            "xmin": ann[1],
                            "ymin": ann[2],
                            "xmax": ann[3],
                            "ymax": ann[4]
                        }
                    )
        
        return annotations
    
    def build_frame_name(self, path, frame_number, frames_names_list):
        frame_idx = frame_number
        pth = os.path.join(path, frames_names_list[frame_idx])
        return pth

    def __format_bbox__(self, bbox):
        """
        Format a tube bbox: [x1,y1,x2,y2] to a correct format
        """
        (width, height) = self.cfg.SHAPE
        bbox = bbox[0:4]
        bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
        # bbox = np.insert(bbox[0:4], 0, id).reshape(1,-1).astype(float)
        bbox = bbox.reshape(1,-1).astype(float)
        if self.cfg.BOX_AS_TENSOR:
            bbox = torch.from_numpy(bbox).float()
        return bbox

    def load_input_1(self, path, frames_indices, frames_names_list, sampled_tube):
        # print('\nload_input_1--> frames_paths')
        tube_images = []
        raw_clip_images = []
        tube_images_t = None
        tube_boxes = []
        if self.transformations['input_1'].itype=='rgb':
            frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames_indices]
            # for j, fp in enumerate(frames_paths):
            #     print(j, ' ', fp)
            for i in frames_paths:
                img = imread(i)
                tube_images.append(img)
                _, frame_name = os.path.split(i)
                
                try:
                    box_idx = sampled_tube['frames_name'].index(frame_name)
                except Exception as e:
                    print("\nOops!", e.__class__, "occurred.")
                    print("sampled_tube['frames_name']: {}, frame: {} , sampled_indices: {}, path: {}".format(sampled_tube['frames_name'], frame_name, frames_indices, path))
                    exit()
                tube_boxes.append(box_idx)
            
            tube_boxes = [sampled_tube['boxes'][b] for b in tube_boxes]
            tube_boxes = [self.__format_bbox__(t) for t in tube_boxes]
            
            # print('\tube_boxes: ', tube_boxes, len(tube_boxes))
            # print('\t tube_images: ', type(tube_images), type(tube_images[0]))
            raw_clip_images = tube_images.copy()
            if self.transformations['input_1'].spatial_transform:
                tube_images_t, tube_boxes_t, t_combination = self.transformations['input_1'].spatial_transform(tube_images, tube_boxes)
       
        return tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images, t_combination

    def get_tube_data(self, tubes_):
        path = self.path
        frames_names_list = os.listdir(path)
        frames_names_list = natural_sort(frames_names_list)
        sampled_frames_indices, chosed_tubes = self.sampler(tubes_)

        # for i in range(len(sampled_frames_indices)):
        #     print('\ntube[{}] \n (1)frames_names_list: {}, \n(2)tube frames_name: {}, \n(3)sampled_frames_indices: {}'.format(i,frames_names_list, chosed_tubes[i]['frames_name'], sampled_frames_indices[i]))
        # print('sampled_frames_indices: ', sampled_frames_indices)
        # print('boxes_from_sampler: ', boxes, boxes[0].shape)
        video_images = []
        video_images_raw = []
        final_tube_boxes = []
        num_tubes = len(sampled_frames_indices)
        for frames_indices, sampled_tube in zip(sampled_frames_indices, chosed_tubes):
            # print('\nload_input_1 args: ', path, frames_indices, boxes)
            tube_images_t, tube_boxes_t, tube_boxes, tube_raw_clip_images, t_combination = self.load_input_1(path, frames_indices, frames_names_list, sampled_tube)
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
                # print(' c_box: ', c_box, c_box.size(), ' index: ', m)
                if tube_box.size(0)==0:
                    print(' Here error: ', path, '\n',
                            tube_box, '\n', 
                            sampled_tube, '\n', 
                            frames_indices, '\n', 
                            tube_boxes_t, len(tube_boxes_t), '\n', 
                            tube_boxes, len(tube_boxes), '\n',
                            t_combination)
                    exit()
                f_box = torch.cat([id_tensor , tube_box], dim=1).float()
            elif self.cfg.BOX_STRATEGY == UNION_BOX:
                all_boxes = [torch.from_numpy(t) for i, t in enumerate(tube_boxes)]
                all_boxes = torch.stack(all_boxes, dim=0).squeeze()
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
        if self.transformations['input_2'] is not None:
            for k in range(len(video_images_raw)):
                if self.cfg.KEYFRAME_STRATEGY == DYNAMIC_IMAGE_KEYFRAME:
                    # key_frame, _ = self.load_input_2_di(sampled_frames_indices[k], path, frames_names_list)
                    key_frame = self.dynamic_image_fn(video_images[k])
                    if self.transformations['input_2'].spatial_transform:
                        key_frame = self.transformations['input_2'].spatial_transform(key_frame)
                else:
                    if self.cfg.KEYFRAME_STRATEGY == RGB_MIDDLE_KEYFRAME:
                        m = int(video_images[k].size(0)/2) #using frames loaded from 3d branch
                        key_frame = video_images[k][m] #tensor 
                        key_frame = key_frame.numpy()
                        if self.transformations['input_2'].spatial_transform:
                            key_frame = self.transformations['input_2'].spatial_transform(key_frame)
                    else:
                        #TODO
                        print('Not implemented yet...')
                        exit()
                key_frames.append(key_frame)
        
        #padding
        # if len(video_images)<self.cfg.NUM_TUBES:
        #     for i in range(self.cfg.NUM_TUBES-len(video_images)):
        #         video_images.append(video_images[len(video_images)-1])
        #         p_box = tube_boxes[len(tube_boxes)-1]
        #         tube_boxes.append(p_box)
        #         if self.transformations['input_2'] is not None:
        #             key_frames.append(key_frames[-1])

        final_tube_boxes = torch.stack(final_tube_boxes, dim=0).squeeze()
        
        if len(final_tube_boxes.shape)==1:
            final_tube_boxes = torch.unsqueeze(final_tube_boxes, dim=0)
            # print('boxes unsqueeze: ', boxes)
        
        video_images = torch.stack(video_images, dim=0).permute(0,4,1,2,3)#.permute(0,2,1,3,4)
        if self.transformations['input_2'] is not None:
            key_frames = torch.stack(key_frames, dim=0)
            if torch.isnan(key_frames).any().item():
                print('Detected Nan at: ', path)
            if torch.isinf(key_frames).any().item():
                print('Detected Inf at: ', path)
            # print('video_images: ', video_images.size())
            # print('key_frames: ', key_frames.size())
            # print('final_tube_boxes: ', final_tube_boxes,  final_tube_boxes.size())
            # print('label: ', label)
            return final_tube_boxes, video_images, self.label, num_tubes, path, key_frames
        else:
            return final_tube_boxes, video_images, self.label, num_tubes, path

    def __getitem__(self, index):
        clip = self.clips[index]
        # image_names, images = self.load_frames(clip)
        image_names = os.listdir(self.path)
        image_names = natural_sort(image_names)
        image_names = list(itemgetter(*clip)(image_names))
        gt = self.load_sp_annotations(image_names, self.sp_annotation)
        
        return clip, image_names, gt, len(clip)
