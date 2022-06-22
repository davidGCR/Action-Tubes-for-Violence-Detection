import os
import random
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import traceback

from utils.dataset_utils import read_JSON_ann, imread
from utils.utils import natural_sort
from utils.tube_utils import JSON_2_videoDetections, JSON_2_tube, tube_2_JSON
from utils.global_var import *
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from tubes.run_tube_gen import extract_tubes_from_video
from datasets.create_tube_sampler import get_sampler

from transformations.dynamic_image_transformation import DynamicImage

def crop(frames, start, size, stride):
    # todo more efficient
    # padding by loop
    while start + (size - 1) * stride > frames[-1]:
        frames *= 2
    
    ed = start + (size - 1) * stride + 1

    start_idx = start-frames[0]
    end_idx = ed-frames[0]
    crop_res = frames[start_idx: end_idx : stride]
    # return frames[start:start + (size - 1) * stride + 1:stride]
    return crop_res

class ClipDataset(data.Dataset):
    """Dataset to load one clip per instance (violence/non-vilolence) in long videos.
    Args:
        cfg (yaml): cfg.TUBE_DATASET
        seq_len (int): Temporal window length, clip length.
        stride (int): Number of frames to skip into a clip.
        make_fn (func): Function to get list of paths, labels, etc.
        random (bool): Flac to extract randomly the clip in the instance.
        transforms (dict): Dictionary with tranformations for the two branches.
        train_set (bool): Flac to indicate to the tube extractor to use the motion map when there is no frames into a clip.
    """
    def __init__(self, cfg, tube_folder, seq_len, stride, make_fn, random, transforms, train_set):
        self.cfg = cfg
        self.train_set = train_set
        self.seq_len = seq_len
        self.stride = stride
        self.paths, self.labels, indices, self.tmp_annotations, self.pers_annotations, self.clips = make_fn()
        self.random = random
        self.transforms = transforms
        self.tube_folder = tube_folder
        self.sampler = get_sampler(self.cfg, self.train_set)
        if self.transforms['input_2'].itype == DYN_IMAGE:
            self.dynamic_image_fn = DynamicImage()

    def sampling(self, clip):
        raw_clip = clip.copy()
        if self.random:
            b = clip[0]
            # e = max(clip[0],clip[-1] - 1 - (self.seq_len - 1) * self.stride)
            e = max(clip[0],clip[-1] - (self.seq_len - 1) * self.stride)
            start = random.randint(b, e)
            crop_r = crop(raw_clip, start, self.seq_len, self.stride)
        else: #center crop
            b = clip[0]
            e = clip[0] + (clip[-1]-clip[0])// 2 - self.seq_len * self.stride // 2
            start = max(b, e)
            crop_r = crop(raw_clip, start, self.seq_len, self.stride)
        
        assert len(crop_r) == self.seq_len, print("Error sampling clip from instance \nclip: {}, \ncrop_r={}/{}, \nstart: {}".format(clip, crop_r, len(crop_r), start))
        return crop_r, start, crop_r[-1]
    
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
    
    def load_clip(self, video_path, sampled_clip, clip):
        """Get a list of frame paths.

        Args:
            video_path (str): Path of the video (folder of frames)
            sampled_clip (list): List of indices corresponding to the clip
            clip (list): Long clip in the video

        Returns:
            list: Paths of the clip
        """
        frames_names = natural_sort([f for f in os.listdir(video_path) if '.jpg' in f])
        # clip_frames = [frames_names[i-1] for i in range(len(frames_names)) if i in sampled_clip]
        try:
            clip_frames = [frames_names[i-1] for i in sampled_clip]
        except Exception as e:
            print("Error reading clip. {}\nvideo: {}\nclip: {}\nsampled_clip: {}".format(e, video_path, clip, sampled_clip))
            # print('\nvideo:', video_path)
            # print('\nsampled_clip:', sampled_clip)
            traceback.print_exc()

        return clip_frames
    
    def load_3D_input(self, path, frames_indices, sampled_tube):
        """Load into memory images anb boxes from a given tube

        Args:
            path (str): Path of the video where the images are stored.
            frames_indices (list): List of integers corresponding to indices of frames to be loaded.
            sampled_tube (dict): Tube in dictionary format.

        Returns:
            (list, list): tube_images_t, tube_boxes_t
        """
        frames_names_list = natural_sort(os.listdir(path))
        tube_images = []
        tube_boxes = []
        frames_paths = [os.path.join(path,frames_names_list[i]) for i in frames_indices]
        # for j, fp in enumerate(frames_paths):
        #     print(j, ' ', fp)
        for i in frames_paths:
            img = imread(i)
            tube_images.append(img)
            _, frame_name = os.path.split(i)
            try:
                box_idx = sampled_tube['frames_name'].index(frame_name)
            except Exception as e:
                print("\nOops! not box_id in tube, ", e.__class__)
                print("\nno filled: {} \nsampled_tube['frames_name']: {} \n frame: {} \n sampled_indices: {} \npath: {}".format(sampled_tube['without_fill_gap'], sampled_tube['frames_name'], frame_name, frames_indices, path))
                exit()
            tube_boxes.append(box_idx)
        
        tube_boxes_raw_size = [sampled_tube['boxes'][b] for b in tube_boxes]
        tube_boxes = [np.array(t[0:4]).reshape(1,-1).astype(float) for t in tube_boxes_raw_size]
        tube_boxes = [np.where(t<0, 0, t).reshape(1,-1).astype(float) for t in tube_boxes]
        
        # print('\ntube_boxes: ', tube_boxes)
        if self.transforms['input_1'].spatial_transform:
            #TODO change the negative coordinates to 0 after the transformation
            tube_images_t, tube_boxes_t, t_combination = self.transforms['input_1'].spatial_transform(tube_images, tube_boxes.copy())
            return tube_images_t, tube_boxes_t

    def get_tube_box(self, tube_boxes_t):
        """Extract one box from all boxes in a tube.

        Args:
            tube_boxes_t (list): List of transformed bboxes

        Raises:
            Exception: Error in tube box if there is an error.

        Returns:
            tensor: box from tube.
        """
        if self.cfg.BOX_STRATEGY == MIDDLE_BOX:
            m = int(len(tube_boxes_t)/2) #middle box from tube
            ##setting id to box
            tube_box = tube_boxes_t[m]
            id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
            if tube_box.size(0)==0:
                raise Exception("Error in tube box: {}/{}".format(tube_boxes_t, len(tube_boxes_t)))
            f_box = torch.cat([id_tensor , tube_box], dim=1).float()
        elif self.cfg.BOX_STRATEGY == UNION_BOX:
            all_boxes = [t for i, t in enumerate(tube_boxes_t)]
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
        return f_box
    
    def load_2D_input(self, video_images, video_boxes):
        """Process the 3D input to get the 2D branch input. Process k inputs, where k is the number of tubes per video.

        Args:
            video_images (list): List of k tensors of shape [t,h,w,3] where t is the number of frames per tube.
            video_boxes (list): List of k tensors of shape [1,5].

        Returns:
            list: List of k tensors of shape [3,224,224].
        """
        key_frames = []
        for k in range(len(video_images)):
            if self.cfg.KEYFRAME_STRATEGY == DYNAMIC_IMAGE_KEYFRAME:
                key_frame = self.dynamic_image_fn(video_images[k])
                # key_frames_raw.append(key_frame)
                if self.cfg.KEYFRAME_CROP:
                    rect = video_boxes[k][1:5]
                    key_frame = key_frame.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                    key_frame = key_frame.resize((self.cfg.SHAPE[0], self.cfg.SHAPE[1]))
                # key_frames_raw.append(key_frame)
                if self.transforms['input_2'].spatial_transform:
                    key_frame = self.transforms['input_2'].spatial_transform(key_frame)
                # key_frames_raw.append(transforms.ToPILImage()(key_frame))
            elif self.cfg.KEYFRAME_STRATEGY == RGB_MIDDLE_KEYFRAME:
                m = int(video_images[k].size(0)/2) #using frames loaded from 3d branch
                key_frame = video_images[k][m] #tensor torch.Size([224, 224, 3])
                #TODO multiply per 255????
                key_frame_pil = transforms.ToPILImage()(key_frame.permute(2,0,1)) #(224, 224)
                # key_frames_raw.append(key_frame_pil)
                if self.cfg.KEYFRAME_CROP:
                    rect = video_boxes[k][1:5]
                    key_frame = key_frame_pil.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                    key_frame = key_frame.resize((self.cfg.SHAPE[0], self.cfg.SHAPE[1]))
                else:
                    key_frame = key_frame_pil
                
                if self.transforms['input_2'].spatial_transform:
                    key_frame = self.transforms['input_2'].spatial_transform(key_frame) #torch.Size([3, 224, 224])
                    # print('key_frame transformed: ', key_frame.size())
                # key_frames_raw.append(transforms.ToPILImage()(key_frame))
            else:
                #TODO
                print('Keyframe option not implemented yet...')
                exit()
            key_frames.append(key_frame)
        return key_frames

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        tmp_annotation = self.tmp_annotations[index]
        pers_annotation = self.pers_annotations[index]
        # frame_rate = tmp_annotation["frame_rate"]
        # clip_start = round(tmp_annotation["segment"][0]*frame_rate)
        # clip_start = 1 if clip_start == 0 else clip_start
        # clip_end = round(tmp_annotation["segment"][1]*frame_rate)
        # clip = np.arange(clip_start, clip_end+1).tolist()
        clip = self.clips[index]
        sampled_clip = []
        if len(clip) > self.seq_len:
            sampled_clip, s, e = self.sampling(clip.copy())
        elif len(clip) == self.seq_len:
            sampled_clip = clip
        # elif len(clip)<self.seq_len:
        #     sampled_clip = np.linspace(clip_start, clip_end, self.seq_len).astype(int).tolist()
        #     s = clip_start
        #     e = clip_end
        clip_frames = self.load_clip(path, sampled_clip, clip) #['frame__001.jpg, ..., frame__016.jpg']
        # print("\nINSTANCE [s:{},e:{}]: {}/{} \nSAMPLED_CLIP: {}/[s,e]={}, \nCLIP_FRAMES before tube extraction: {}".format(clip_start, clip_end, clip, len(clip), sampled_clip, (s,e), clip_frames))
        try:
            sampled_clip_indices = [i-1 for i in sampled_clip]
            tubes = self.extract_tubes(path, pers_annotation, sampled_clip_indices, clip_frames, label)
        except Exception as e:
            print("\nOops! Extract tube exception: ", e.__class__, "occurred.\n", e)
            print("Extract tube parameters \npers_annot: {}\nclip: {}\nsampled clip: {}\nsampled_clip_indices: {}\nclip_frames: {}\nlabel: {}".format(pers_annotation, clip, sampled_clip, sampled_clip_indices, clip_frames, label))
            traceback.print_exc()
            # exit()
        # print("\ntubes: ", len(tubes))
        sampled_frames_indices, chosed_tubes = self.sampler(tubes)
        # print("\nsampled_frames_indices:", sampled_frames_indices)
        # print("\nchosed_tubes:", chosed_tubes)
        
        
        video_images = []
        video_boxes = []
        for frames_indices, sampled_tube in zip(sampled_frames_indices, chosed_tubes):
            tube_images_t, tube_boxes_t = self.load_3D_input(path, frames_indices, sampled_tube)
            video_images.append(torch.stack(tube_images_t, dim=0))
            #get one box per tube
            try:
                tube_box = self.get_tube_box(tube_boxes_t)
            except Exception as e:
                print("Error extracting tube box. ", e)
                traceback.print_exc()
            video_boxes.append(tube_box)
        keyframes = self.load_2D_input(video_images, video_boxes)
        video_images = torch.stack(video_images, dim=0).permute(0,4,1,2,3)
        video_boxes = torch.stack(video_boxes, dim=0).squeeze()
        keyframes = torch.stack(keyframes, dim=0)
        # return path, label, tmp_annotation, pers_annotation, clip, sampled_clip, video_images, video_boxes, keyframes
        return video_boxes, video_images, label, path, keyframes

def load3DInput(path, frames_indices, sampled_tube, spatial_transform):
    """Load into memory images anb boxes from a given tube

    Args:
        path (str): Path of the video where the images are stored.
        frames_indices (list): List of integers corresponding to indices of frames to be loaded.
        sampled_tube (dict): Tube in dictionary format.
        spatial_transform (transform): Spatial transformation

    Returns:
        (list, list): tube_images_t, tube_boxes_t
    """
    frames_names_list = natural_sort(os.listdir(path))
    tube_images = []
    tube_boxes = []
    frames_paths = [os.path.join(path,frames_names_list[i]) for i in frames_indices]
    # for j, fp in enumerate(frames_paths):
    #     print(j, ' ', fp)
    for i in frames_paths:
        img = imread(i)
        tube_images.append(img)
        _, frame_name = os.path.split(i)
        try:
            box_idx = sampled_tube['frames_name'].index(frame_name)
        except Exception as e:
            print("\nOops! not box_id in tube, ", e.__class__)
            print("\nno filled: {} \nsampled_tube['frames_name']: {} \n frame: {} \n sampled_indices: {} \npath: {}".format(sampled_tube['without_fill_gap'], sampled_tube['frames_name'], frame_name, frames_indices, path))
            exit()
        tube_boxes.append(box_idx)
    
    tube_boxes_raw_size = [sampled_tube['boxes'][b] for b in tube_boxes]
    tube_boxes = [np.array(t[0:4]).reshape(1,-1).astype(float) for t in tube_boxes_raw_size]
    tube_boxes = [np.where(t<0, 0, t).reshape(1,-1).astype(float) for t in tube_boxes]
    
    # print('\ntube_boxes: ', tube_boxes)
    if spatial_transform:
        #TODO change the negative coordinates to 0 after the transformation
        tube_images_t, tube_boxes_t, t_combination = spatial_transform(tube_images, tube_boxes.copy())
        return tube_images_t, tube_boxes_t

def load2DInput(video_images, 
                  video_boxes, 
                  dynamic_image_fn, 
                  keyframe_str,
                  spatial_transform,
                  shape,
                  keyframe_crop):
    """Process the 3D input to get the 2D branch input. Process k inputs, where k is the number of tubes per video.

    Args:
        video_images (list): List of k tensors of shape [t,h,w,3] where t is the number of frames per tube.
        video_boxes (list): List of k tensors of shape [1,5]
        dynamic_image_fn (function): Function to build dynamic image
        keyframe_str (str): Keyframe sampling strategy
        spatial_transform (transformation): Spatial transformation
        shape (array, tuple): Shape to resize input
        keyframe_crop (bool): Flac to indicate if input would be crop

    Returns:
        list: List of k tensors of shape [3,224,224].
    """
    key_frames = []
    for k in range(len(video_images)):
        if keyframe_str == DYNAMIC_IMAGE_KEYFRAME:
            key_frame = dynamic_image_fn(video_images[k])
            # key_frames_raw.append(key_frame)
            if keyframe_str:
                rect = video_boxes[k][1:5]
                key_frame = key_frame.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                key_frame = key_frame.resize((shape[0], shape[1]))
            # key_frames_raw.append(key_frame)
            if spatial_transform:
                key_frame = spatial_transform(key_frame)
            # key_frames_raw.append(transforms.ToPILImage()(key_frame))
        elif keyframe_str == RGB_MIDDLE_KEYFRAME:
            m = int(video_images[k].size(0)/2) #using frames loaded from 3d branch
            key_frame = video_images[k][m] #tensor torch.Size([224, 224, 3])
            #TODO multiply per 255????
            key_frame_pil = transforms.ToPILImage()(key_frame.permute(2,0,1)) #(224, 224)
            # key_frames_raw.append(key_frame_pil)
            if keyframe_crop:
                rect = video_boxes[k][1:5]
                key_frame = key_frame_pil.crop((rect[0].item(),rect[1].item(),rect[2].item(),rect[3].item()))
                key_frame = key_frame.resize((shape[0], shape[1]))
            else:
                key_frame = key_frame_pil
            
            if spatial_transform:
                key_frame = spatial_transform(key_frame) #torch.Size([3, 224, 224])
                # print('key_frame transformed: ', key_frame.size())
            # key_frames_raw.append(transforms.ToPILImage()(key_frame))
        else:
            #TODO
            print('Keyframe option not implemented yet...')
            exit()
        key_frames.append(key_frame)
    return key_frames

def getTubeBox(tube_boxes_t, box_strategy):
    """Extract one box from all boxes in a tube.

    Args:
        tube_boxes_t (list): List of transformed bboxes
        box_strategy (str): Type of sample for bbox

    Raises:
        Exception: Error in tube box if there is an error.

    Returns:
        tensor: box from tube.
    """
    if box_strategy == MIDDLE_BOX:
        m = int(len(tube_boxes_t)/2) #middle box from tube
        ##setting id to box
        tube_box = tube_boxes_t[m]
        id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
        if tube_box.size(0)==0:
            raise Exception("Error in tube box: {}/{}".format(tube_boxes_t, len(tube_boxes_t)))
        f_box = torch.cat([id_tensor , tube_box], dim=1).float()
    elif box_strategy == UNION_BOX:
        all_boxes = [t for i, t in enumerate(tube_boxes_t)]
        all_boxes = torch.stack(all_boxes, dim=0).squeeze()
        mins, _ = torch.min(all_boxes, dim=0)
        x1 = mins[0].unsqueeze(dim=0).float()
        y1 = mins[1].unsqueeze(dim=0).float()
        maxs, _ = torch.max(all_boxes, dim=0)
        x2 = maxs[2].unsqueeze(dim=0).float()
        y2 = maxs[3].unsqueeze(dim=0).float()
        id_tensor = torch.tensor([0]).float()
        f_box = torch.cat([id_tensor , x1, y1, x2, y2]).float()
    elif box_strategy == ALL_BOX:
        f_box = [torch.cat([torch.tensor([i]).unsqueeze(dim=0), torch.from_numpy(t)], dim=1).float() for i, t in enumerate(tube_boxes)]
        f_box = torch.stack(f_box, dim=0)
    return f_box
        
class SequentialDataset(data.Dataset):
    """Load a long video sequentially

    Args:
        cfg (yaml): cfg.TUBE_DATASET
        seq_len (int): Temporal window length, clip length.
        video_path (str): Path to the folder with frames
        tubes_path (str): [description]
        annotations (str): [description]
        pers_detect_annot (str): [description]
        frame_rate (int): [description]
        transform (torch transform): [description]
    """
    def __init__(self, 
                 cfg,
                 seq_len, 
                 video_path, 
                 tubes_path, 
                 annotations, 
                 pers_detect_annot, 
                 frame_rate, 
                 transforms):
        self.cfg = cfg
        self.seq_len = seq_len
        self.tubes_path = tubes_path
        self.video_path = video_path
        self.annotations = annotations
        self.pers_detect_annot = pers_detect_annot
        self.frame_rate = frame_rate
        self.frames = natural_sort([f for f in os.listdir(self.video_path) if '.jpg' in f])
        
        indices = [x for x in range(0, len(self.frames), 1)]
        self.sequences = [indices[x:x+self.seq_len] for x in range(0, len(self.frames), self.seq_len)]
        
        self.max_overlap = round(0.5*self.seq_len)
        self.transforms = transforms
        self.sampler = get_sampler(self.cfg, False)
        if self.transforms['input_2'].itype == DYN_IMAGE:
            self.dynamic_image_fn = DynamicImage()

    def seq2frames(self, sequence):
        frames = [self.frames[idx] for idx in sequence]
        return frames

    def padding(self, sequence, index):
        previous_sequence = self.sequences[index-1]
        pad_len = self.seq_len - len(sequence)
        pad_sequence = previous_sequence[-pad_len:]
        union_seq = pad_sequence + sequence
        return union_seq
    
    def load_tubes(self, sequence, sequence_video_names):
        video_name = self.video_path.split('/')[-1]
        tube_path = os.path.join(self.tubes_path, "{}_from_{}_to_{}.json".format(video_name, sequence[0]+1, sequence[-1]+1))
        # print('tube_path: ', tube_path)
        if not os.path.isfile(tube_path):
            print('Extracting/saving tubes at: ', tube_path)
            person_detections = JSON_2_videoDetections(self.pers_detect_annot)
            TUBE_BUILD_CONFIG['person_detections'] = person_detections
            tubes, time = extract_tubes_from_video(sequence, sequence_video_names, MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG, None)
            tube_2_JSON(tube_path, tubes)
        else:
            tubes = JSON_2_tube(tube_path)
        return tubes
    
    def label(self, sequence):
        s = sequence[0]+1
        e = sequence[-1]+1
        label = 0
        annotation = [-1, -1]
        for an in self.annotations:
            gt_s = round(an["segment"][0]*self.frame_rate)
            gt_e = round(an["segment"][1]*self.frame_rate)
            # print(an, gt_s, gt_e, type(s))
            if s >= gt_s and e <= gt_e:
                label = 1
                annotation = an, gt_s, gt_e
                break
            if s+self.max_overlap >= gt_s and e <= gt_e:
                label = 1
                annotation = an, gt_s, gt_e
                break
            if s >= gt_s and e-self.max_overlap <= gt_e:
                label = 1
                annotation = an, gt_s, gt_e
                break
        return label, annotation

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        if len(sequence)<self.seq_len:
            sequence = self.padding(sequence, index)
        label, annotation = self.label(sequence)
        frames_names = self.seq2frames(sequence)
        # print('sequence {}/{}/names: {}'.format(index, sequence, len(frames_names)))

        
        tubes = self.load_tubes(sequence, frames_names)
        # print('sequence tubes: ', len(tubes))
        video_images = []
        video_boxes = []
        ntubes = len(tubes)
        if ntubes>0:
            sampled_frames_indices, chosed_tubes = self.sampler(tubes)
            # print("\nsampled_frames_indices:", sampled_frames_indices)
            # print("\nchosed_tubes:", chosed_tubes)
            for frames_indices, sampled_tube in zip(sampled_frames_indices, chosed_tubes):
                tube_images_t, tube_boxes_t = load3DInput(self.video_path, 
                                                          frames_indices, 
                                                          sampled_tube,
                                                          self.transforms['input_1'].spatial_transform)
                video_images.append(torch.stack(tube_images_t, dim=0))
                #get one box per tube
                try:
                    tube_box = getTubeBox(tube_boxes_t, self.cfg.BOX_STRATEGY)
                except Exception as e:
                    print("Error extracting tube box. ", e)
                    traceback.print_exc()
                video_boxes.append(tube_box)
                # print('tube_box: ', tube_box, tube_box.size())
            keyframes = load2DInput(video_images, 
                                    video_boxes, 
                                    self.dynamic_image_fn, 
                                    self.cfg.KEYFRAME_STRATEGY, 
                                    self.transforms['input_2'].spatial_transform,
                                    self.cfg.SHAPE, 
                                    self.cfg.KEYFRAME_CROP)
        else:
            # transform = transforms.ToTensor()
            transform = transforms.Compose([
                # transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            tube_images_t = []
            for fn in frames_names:
                img = imread(os.path.join(self.video_path, fn))
                tube_images_t.append(transform(img))
            tube_images_t = tube_images_t[0:16]
            keyframes = [tube_images_t[0].clone()]
            keyframes = self.cfg.NUM_TUBES*keyframes
            tube_images_t = torch.stack(tube_images_t, dim=0).permute(0,2,3,1)
            video_images = self.cfg.NUM_TUBES*[tube_images_t]
            video_boxes = self.cfg.NUM_TUBES*[torch.tensor([0, 10, 10, 40, 40])]
            
            
        video_images = torch.stack(video_images, dim=0).permute(0,4,1,2,3)
        keyframes = torch.stack(keyframes, dim=0)
        video_boxes = torch.stack(video_boxes, dim=0).squeeze()
        # print('video_images from dataset: ', video_images.size())
        # return sequence, frames_names, video_images, label, annotation
        # return video_boxes, video_images, label, tubes, keyframes
        # path = ""
        return video_boxes, video_images, label, ntubes, keyframes


import json

def get_neg_instances(annotations, total_duration, min_dur = 0.5):
    """Extract temporal negative instances from a long video.

    Args:
        annotations (list): List of dicts of positive instance annotations
        total_duration (float): Video duration.
        min_dur (float, optional): Minimum temporal duration for negative instances. Defaults to 0.5.

    Returns:
        list: List of dicts of negative instances.
    """
    negative_instances = []
    for i in range(len(annotations)):
        # print('pos instance: ', annotations[i])
        current_pos_instance = annotations[i]["segment"]
        if i==0:
            s = .0
            e = current_pos_instance[0]
        else:
            previous_pos_instance = annotations[i-1]["segment"]
            s = previous_pos_instance[1]
            e = current_pos_instance[0]
        if e-s >= min_dur:
            negative_instances.append({
                "segment": [s, e],
                "label": "NonFight"
            })
        #last segment
        if i==len(annotations)-1:
            s = current_pos_instance[1]
            e = total_duration
            if e-s >= min_dur:
                negative_instances.append({
                    "segment": [s, e],
                    "label": "NonFight"
                })
    return negative_instances

def add_negative_annot(ann_file):
    """Add negative annotations to ground truth file. The modified annotations are saved in json format

    Args:
        ann_file (str): File path of json format with temporal ground truth
    """
    data = read_JSON_ann(ann_file)
    for i, key in enumerate(data["database"]):
        print(i, key)

        video_gt = data["database"][key]
        neg_segments = get_neg_instances(video_gt["annotations"], video_gt["duration"])
        video_gt["n_annotations"] = neg_segments
    
    with open("groundtruth_modified.json", "w") as outfile:
        json.dump(data, outfile)

# def make_CCTVFights_dataset_clips(root, json_file, pers_annotations_folder, subset='training'):
#     """Load clip paths considering each positive and negative instance as an individual sample

#     Args:
#         root (str): Path of the dataset folder.
#         json_file (str): Annotations ground truth path.
#         subset (str, optional): Subset to load. Defaults to 'training'.

#     Returns:
#         (list, list, list, list, list) : Lists of paths, labels, indices, tmp_annotations = [{'segment': [142.2, 230.68], 'label': 'NonFight', 'frame_rate': 25.0}, ...], pers_annotation
#     """
#     data = read_JSON_ann(json_file)
#     paths = []
#     labels = []
#     indices = []
#     tmp_annotations = []
#     pers_annotations = []
#     for i, key in enumerate(data["database"]):
#         if data["database"][key]["subset"] == subset:
#             for j in range(len(data["database"][key]["annotations"])):
#                 paths.append(os.path.join(root, key))
#                 labels.append(1)
#                 indices.append(j)

#                 temp_annot = data["database"][key]["annotations"][j]
#                 temp_annot["frame_rate"] = data["database"][key]["frame_rate"]
#                 tmp_annotations.append(temp_annot)
#                 pers_annotations.append(os.path.join(pers_annotations_folder, key+'.json'))
#             for k in range(len(data["database"][key]["n_annotations"])):
#                 paths.append(os.path.join(root, key))
#                 labels.append(0)
#                 indices.append(k)

#                 temp_annot = data["database"][key]["n_annotations"][k]
#                 temp_annot["frame_rate"] = data["database"][key]["frame_rate"]
#                 tmp_annotations.append(temp_annot)
#                 pers_annotations.append(os.path.join(pers_annotations_folder, key+'.json'))
#     return paths, labels, indices, tmp_annotations, pers_annotations

# def make_CCTVFights_dataset(root, root_person_detec, json_file, subset='testing'):
#     """Load clip paths considering long videos.

#     Args:
#         root (str): Path of the dataset folder.
#         root_person_detec (str): Path of the person detections folder.
#         json_file (str): Annotations ground truth path.
#         subset (str, optional): Subset to load. Defaults to 'testing'.

#     Returns:
#         (list, list, list, list) : Lists of paths, frame_rates, tmp_annotations, person_det_files
#     """
#     data = read_JSON_ann(json_file)
#     paths = []
#     frame_rates = []
#     tmp_annotations = []
#     person_det_files = []
#     for i, key in enumerate(data["database"]):
#         if data["database"][key]["subset"] == subset:
#             paths.append(os.path.join(root, key))
#             frame_rates.append(data["database"][key]["frame_rate"])
#             tmp_annotations.append(data["database"][key]["annotations"])
#             person_det_files.append(os.path.join(root_person_detec, 'fights', key+'.json'))
#     return paths, frame_rates, tmp_annotations, person_det_files

