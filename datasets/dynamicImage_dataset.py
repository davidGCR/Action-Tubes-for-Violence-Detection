import os
import torch
import torch.utils.data as data
from transformations.dynamic_image_transformation import DynamicImage
# from temporal_sampling import CenterCrop, RandomCrop, SegmentsCrop
from utils.utils import natural_sort
from datasets.temporal_sampling import *
from operator import itemgetter

class DynamicImageDataset(data.Dataset):
    def __init__(self, cfg, make_fn, train_set, transform):
        self.cfg = cfg
        self.make_fn = make_fn
        self.train_set = train_set
        self.transform = transform
        
        self.paths, self.labels, self.annotations = self.make_fn()
        if self.train_set:
            self.temp_sampler = RandomCrop(self.cfg.CLIP_LEN, self.cfg.CLIP_STRIDE, 'rgb')
        else:
            self.temp_sampler = CenterCrop(self.cfg.CLIP_LEN, self.cfg.CLIP_STRIDE, 'rgb')
        
        self.dynamic_image_fn = DynamicImage()
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        frames_list = natural_sort(os.listdir(path)) #['frame001.jpg', 'frame002.jpg', 'frame003.jpg', ...]
        num_frames = len(frames_list)
        # print(frames_list, num_frames)
        
        indices = list(range(1, num_frames ))
        clip_o_clips = self.temp_sampler(indices) #[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]
        clip_o_clips = list(itemgetter(*clip_o_clips)(frames_list)) #['frame008.jpg', 'frame010.jpg', 'frame012.jpg', ..., 'frame038.jpg']
        clip_o_clips = [os.path.join(path, n) for n in clip_o_clips if os.path.isfile(os.path.join(path, n))] #['C:\\Users\\David\\Desktop\\DATASETS\\HockeyFightsDATASET/frames\\nonviolence\\303\\frame005.jpg', 'C:\\Users\\David\\Desktop\\DATASETS\\HockeyFightsDATASET/frames\\nonviolence\\303\\frame007.jpg', ...]
        
        # print(clip_o_clips, len(clip_o_clips))
        
        dynamic_image = self.dynamic_image_fn(clip_o_clips) #<class 'PIL.Image.Image'> (360, 288)
        if self.transform:
            dynamic_image = self.transform(dynamic_image) #<class 'torch.Tensor'> torch.Size([3, 224, 224])
        return dynamic_image, label