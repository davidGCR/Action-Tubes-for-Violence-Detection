from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from transformations.dynamic_image_transformation import DynamicImage
from tubes.plot_tubes import plot_tubes
from tubes.run_tube_gen import extract_tubes_from_video
from utils.dataset_utils import imread
from utils.global_var import *
from utils.tube_utils import JSON_2_tube, JSON_2_videoDetections, tube_2_JSON
from utils.utils import  natural_sort
import cv2

class VideoDemo(data.Dataset):
    def __init__(self, 
                 cfg,
                 path: str, 
                 tub_cfg: dict, 
                 mot_cgf: dict, 
                 tub_file=None, 
                 ped_file=None, 
                 vizualize_tubes:bool=False,
                 save_folder=None,
                 vizualize_keyframe:bool=False,
                 transformations=None):
        """Dataset to load/extract tubes from a single video

        Args:
            cfg (Yaml): cfg.TUBE_DATASET.
            path (str): Path to video folder with frames.
            tub_cfg (dict): Configuration for online tube generation. Obligatory to vizualization.
            mot_cgf (dict): Configuration for online motion segmentation.
            tub_file (_type_, optional): Path to json file with precomputed tubes. Defaults to None.
            ped_file (_type_, optional): Path to json file with person detections. Defaults to None.
            vizualize_tubes (bool, optional): Flac to vizualize online tube generation. Defaults to False.
            save_folder (str, optional): Folder to save tubes plots. Defaults to None.
            transformations (dict, optional): Spatial transformations for two stream model. Defaults to False.
        """
        self.save_folder = save_folder
        self.cfg = cfg
        self.path = Path(path)
        self.check_file(self.path)
        self.vizualize_tubes = vizualize_tubes
        
        self.tub_file = self.check_file(tub_file)
        self.ped_file = self.check_file(ped_file)
        # self.frames = natural_sort([frame for frame in os.listdir(self.path) if '.jpg' in frame])
        self.frames = natural_sort([p.name for p in self.path.iterdir()])
        self.num_frames = len(self.frames)
        
        #path root
        self.root = self.path.parents[2]

        # Tube generation config
        self.tub_cfg = tub_cfg
        self.tub_cfg['dataset_root'] = self.root
        self.mot_cgf = mot_cgf
        self.transformations = transformations
        self.vizualize_keyframe = vizualize_keyframe
        
        if vizualize_tubes:
            self.tub_cfg['plot_config']['plot_tubes'] = True
            self.tub_cfg['plot_config']['debug_mode'] = False

        self.tubes = self.gen_tubes()
        print('Generados {} action tubes'.format(len(self.tubes)))
    
    def __len__(self):
        return len(self.tubes)
    
    def __getitem__(self, index):
        tube = self.tubes[index]
        tube_frames_idxs = self.__sampled_tube_frames_indices__(tube['foundAt'], 
                                                                self.cfg.NUM_FRAMES, 
                                                                self.cfg.FRAMES_STRATEGY)
        f_box, tube_images_t, key_frame = get_tube_data(tube, 
                                                        tube_frames_idxs, 
                                                        self.frames, 
                                                        self.path, 
                                                        self.cfg.BOX_STRATEGY, 
                                                        self.transformations, 
                                                        self.cfg.SHAPE,
                                                        self.cfg.KEYFRAME_STRATEGY,
                                                        self.cfg.KEYFRAME_CROP,
                                                        self.vizualize_keyframe,
                                                        DynamicImage(vizualize=False))
        return f_box, tube_images_t, key_frame, tube
    
    def __sampled_tube_frames_indices__(self, 
                                        tube_found_at: list,
                                        tube_len,
                                        sample_strategy):
        max_video_len = tube_found_at[-1]
        if len(tube_found_at) == tube_len: 
            return tube_found_at
        if len(tube_found_at) > tube_len:
            if sample_strategy == MIDDLE_FRAMES:
                n = len(tube_found_at)
                m = int(n/2)
                arr = np.array(tube_found_at)
                centered_array = arr[m-int(tube_len/2) : m+int(tube_len/2)]
            elif sample_strategy == EVENLY_FRAMES:
                min_frame = tube_found_at[0]
                tube_frames_idxs = np.linspace(min_frame, max_video_len, tube_len).astype(int)
                tube_frames_idxs = tube_frames_idxs.tolist()
            return centered_array.tolist()
        if len(tube_found_at) < tube_len: #padding
            min_frame = tube_found_at[0]
            # TODO 
            tube_frames_idxs = np.linspace(min_frame, max_video_len, tube_len).astype(int)
            tube_frames_idxs = tube_frames_idxs.tolist()
            return tube_frames_idxs
        
    def check_file(self, path):
        if path:
            path = Path(path)
            if not path.exists():
                print('ERROR: File: {} does not exist!!!'.format(path))
        return path

    def temporal_step(self):
        # indices = np.linspace(0, self.num_frames, dtype=np.int16).tolist()
        indices = list(range(0, self.num_frames))
        names = [str(self.path/self.frames[i]) for i in indices]
        return indices, names
    
    def plot_best_tube(self, indices_to_plot):
        tubes = JSON_2_tube(str(self.tub_file))
        indices, names = self.temporal_step()
        tp = [tubes[i] for i in indices_to_plot]
        
        plot_tubes(names, tp, save_folder=self.save_folder)
    
    def plot_gen_tubes(self):
        tubes = None
        if self.tub_file:
            print('Loading tubes from: ', self.tub_file)
            tubes = JSON_2_tube(str(self.tub_file))
            indices, names = self.temporal_step()
            plot_tubes(names, tubes, save_folder=None)
                    
    def gen_tubes(self):
        tubes = None
        # if self.tub_file:
        #     print('Loading tubes from: ', self.tub_file)
        #     tubes = JSON_2_tube(str(self.tub_file))
        #     indices, names = self.temporal_step()
        #     if self.vizualize_tubes:
        #         plot_tubes(names, tubes, save_folder=None)
                
        # else:
        print('Extracting tubes...')
        if self.ped_file:
            person_detections = JSON_2_videoDetections(str(self.ped_file))
            self.tub_cfg['person_detections'] = person_detections
            indices, names = self.temporal_step()
            tubes, time = extract_tubes_from_video(indices, names, self.mot_cgf, self.tub_cfg, None)
            if self.save_folder:
                tube_2_JSON(self.save_folder, tubes)
                
            # if self.vizualize_tubes:
            #     plot_tubes(names, tubes, save_folder=None)
        else:
            print('ERROR: No persons detections file!!!')
        
        return tubes

def get_tube_data(tube: dict, 
                  tube_frames_indices: list, 
                  frames_names_list: list, 
                  video_path: str, 
                  box_trategy: str,
                  transformations: dict,
                  shape: tuple,
                  keyframe_strategy: int,
                  keyframe_crop: bool,
                  vizualize_keyframe: bool,
                  dyn_fn: any):
    """Gets tensors from an action tube

    Args:
        tube (dict): Action tube
        tube_frames_indices (list): List of integers corresponding to indices in a list of frames
        frames_names_list (list): List of imgs files.
        video_path (str): Path to the video folder with frames
        box_trategy (str): Type of tube box sampling.
        transformations (dict): Spatial transformations for two stream model.
        shape (tuple): Spatial size.
        keyframe_strategy (int): Type of keyframe from tube,
        dyn_fn (function): Function to compute a dynamic image
    """
    input_1 = load_input_1(video_path, tube_frames_indices, frames_names_list, tube, transformations, shape)
    tube_images_t, tube_boxes_t, tube_boxes, tube_raw_clip_images, t_combination = input_1
    tube_images_t = torch.stack(tube_images_t, dim=0)
    
    #Box extracted from tube
    tube_box = None
    if box_trategy == MIDDLE_BOX:
        m = int(len(tube_boxes)/2) #middle box from tube
        ##setting id to box
        tube_box = tube_boxes_t[m]
        id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
        # print('\n', ' id_tensor: ', id_tensor,id_tensor.size())
        # print(' c_box: ', c_box, c_box.size(), ' index: ', m)
        if tube_box.size(0)==0:
            print('get_tube_data error in tube_box: ', video_path, '\n',
                    tube_box, '\n', 
                    tube, '\n', 
                    tube_frames_indices, '\n', 
                    tube_boxes_t, len(tube_boxes_t), '\n', 
                    tube_boxes, len(tube_boxes), '\n',
                    t_combination)
            exit()
        f_box = torch.cat([id_tensor , tube_box], dim=1).float()
    elif box_trategy == UNION_BOX:
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
    elif box_trategy == ALL_BOX:
        f_box = [torch.cat([torch.tensor([i]).unsqueeze(dim=0), torch.from_numpy(t)], dim=1).float() for i, t in enumerate(tube_boxes)]
        f_box = torch.stack(f_box, dim=0)
    
    f_box = torch.unsqueeze(f_box, dim=0)
    # print('tube_box: ', f_box.size())
    
    #load keyframes
    # key_frames = []
    if transformations['input_2'] is not None:
        if keyframe_strategy == DYNAMIC_IMAGE_KEYFRAME:
            # key_frame, _ = self.load_input_2_di(sampled_frames_indices[k], path, frames_names_list)
            key_frame = dyn_fn(tube_images_t)
            if keyframe_crop:
                rect = f_box[0][1:5]
                key_frame = key_frame.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                key_frame = key_frame.resize((shape[0], shape[1]))
                if vizualize_keyframe:
                    frame = cv2.resize(np.array(key_frame), (600,600))
                    cv2.imshow('Dynamic I', frame)
                    key = cv2.waitKey(0)
                    if key == 27:#if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
            if transformations['input_2'].spatial_transform:
                key_frame = transformations['input_2'].spatial_transform(key_frame)
        else:
            if keyframe_strategy == RGB_MIDDLE_KEYFRAME:
                m = int(tube_images_t.size(0)/2) #using frames loaded from 3d branch
                key_frame = tube_images_t[m] #tensor 
                key_frame = key_frame.numpy()
                if transformations['input_2'].spatial_transform:
                    key_frame = transformations['input_2'].spatial_transform(key_frame)
            else:
                #TODO
                print('Not implemented yet...')
                exit()
    # print('keyframe: ', key_frame.size())
    tube_images_t = tube_images_t.permute(3,0,1,2)#.permute(0,2,1,3,4)
    # print('tube_images_t: ', tube_images_t.size())
    return f_box, tube_images_t, key_frame

def load_input_1(path:str, frames_indices:list, frames_names_list:list, sampled_tube:dict, transformations:dict, shape:tuple):
    """Gets tensors from an action tube

    Args:
        path (str): Path to the video folder
        frames_indices (list): _description_
        frames_names_list (list): _description_
        sampled_tube (dict): _description_
        transformations (dict): _description_
        shape (tuple): _description_

    Returns:
        tuple: tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images, t_combination
    """
    # print('\nload_input_1--> frames_paths')
    tube_images = []
    raw_clip_images = []
    tube_images_t = None
    tube_boxes = []
    if transformations['input_1'].itype=='rgb':
        frames_paths = [build_frame_name(path, i, frames_names_list) for i in frames_indices]
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
        tube_boxes = [__format_bbox__(t, shape) for t in tube_boxes]
        
        # print('\tube_boxes: ', tube_boxes, len(tube_boxes))
        # print('\t tube_images: ', type(tube_images), type(tube_images[0]))
        raw_clip_images = tube_images.copy()
        if transformations['input_1'].spatial_transform:
            tube_images_t, tube_boxes_t, t_combination = transformations['input_1'].spatial_transform(tube_images, tube_boxes)
    
    return tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images, t_combination

def build_frame_name(path, frame_number, frames_names_list):
    frame_idx = frame_number
    pth = os.path.join(path, frames_names_list[frame_idx])
    return pth

def __format_bbox__(bbox, shape, box_as_tensor=False):
    """
    Format a tube bbox: [x1,y1,x2,y2] to a correct format
    """
    (width, height) = shape
    bbox = bbox[0:4]
    bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
    # bbox = np.insert(bbox[0:4], 0, id).reshape(1,-1).astype(float)
    bbox = bbox.reshape(1,-1).astype(float)
    if box_as_tensor:
        bbox = torch.from_numpy(bbox).float()
    return bbox
