import os
import glob
from operator import itemgetter
import numpy as np
import json
import re
import random

class MakeImageHMDB51():
    def __init__(self, root, annotation_path, fold, train):
        self.root = root
        self.annotation_path = annotation_path
        self.fold = fold
        self.train = train
        self.TRAIN_TAG = 1
        self.TEST_TAG = 2
    
    def __select_fold__(self, video_list):
        target_tag = self.TRAIN_TAG if self.train else self.TEST_TAG
        split_pattern_name = "*test_split{}.txt".format(self.fold)
        split_pattern_path = os.path.join(self.annotation_path, split_pattern_name)
        annotation_paths = glob.glob(split_pattern_path)
        # for a in annotation_paths:
        #     print(a)
        selected_files = []
        for filepath in annotation_paths:
            with open(filepath) as fid:
                lines = fid.readlines()
            for line in lines:
                video_filename, tag_string = line.split()
                tag = int(tag_string)
                if tag == target_tag:
                    selected_files.append(video_filename[:-4])
        selected_files = set(selected_files)

        # print(selected_files, len(selected_files))
        indices = []
        for video_index, video_path in enumerate(video_list):
            # print(os.path.basename(video_path))
            if os.path.basename(video_path) in selected_files:
                indices.append(video_index)

        return indices

    def __call__(self):
        # classes = sorted(os.listdir(self.root))
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # print("classes:",classes)
        # print("class_to_idx:",class_to_idx)
        paths = []
        labels = []
        for c in classes:
            class_path = os.path.join(self.root, c)
            for v in os.scandir(class_path):
                if v.is_dir():
                    video_path = os.path.join(class_path,v.name)
                    paths.append(video_path)
                    labels.append(class_to_idx[c])

        # print(paths, len(paths))
        indices = self.__select_fold__(paths)
        paths = list(itemgetter(*indices)(paths))
        labels = list(itemgetter(*indices)(labels))
        # print(paths, len(paths))
        # print(labels, len(labels))
        return paths, labels

class MakeHockeyDataset():
    def __init__(self, root, train, cv_split_annotation_path, path_annotations=None):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.cv_split_annotation_path = cv_split_annotation_path
        # self.split = split
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["nonviolence","violence"]
    
    def split(self):
        split = "training" if self.train else "validation"
        return split
    
    def load_annotation_data(self):
        with open(self.cv_split_annotation_path, 'r') as data_file:
            return json.load(data_file)
    
    def get_video_names_and_labels(self, data, split):
        video_names = []
        video_labels = []
        annotations = []

        for key, val in data['database'].items():
            if val['subset'] == split:
                label = val['annotations']['label']
                cl = 'violence' if label=='fi' else 'nonviolence'

                label = 0 if label=='no' else 1
                v_name = re.findall(r'\d+', key)[0]
                folder = os.path.join(self.root, cl, v_name)
                assert os.path.isdir(folder), "Folder:{} does not exist!!!".format(folder)
                video_names.append(folder)
                video_labels.append(label)
                if self.path_annotations:
                    ann_file = os.path.join(self.path_annotations, cl, v_name+'.json')
                    assert os.path.isfile(ann_file), "Annotation file:{} does not exist!!!".format(ann_file)
                    annotations.append(ann_file)

        return video_names, video_labels, annotations
    
    def __call__(self):
        data = self.load_annotation_data()
        split = self.split()
        paths, labels, annotations = self.get_video_names_and_labels(data, split)
        return paths, labels, annotations

class MakeRLVDDataset():
    def __init__(self, root, train, cv_split_annotation_path, path_annotations=None):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.cv_split_annotation_path = cv_split_annotation_path
        # self.split = split
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["NonViolence","Violence"]
    
    def split(self):
        split = "training" if self.train else "validation"
        return split
    
    def load_annotation_data(self):
        with open(self.cv_split_annotation_path, 'r') as data_file:
            return json.load(data_file)
    
    def get_video_names_and_labels(self, data, split):
        video_names = []
        video_labels = []
        annotations = []
        # num_frames = []

        for key, val in data['database'].items():
            if val['subset'] == split:
                label = val['annotations']['label']
                cl = 'Violence' if label=='fi' else 'NonViolence'

                label = 0 if label=='no' else 1
                # v_name = re.findall(r'\d+', key)[0]
                v_name = key
                folder = os.path.join(self.root, cl, v_name)
                assert os.path.isdir(folder), "Folder:{} does not exist!!!".format(folder)
                video_names.append(folder)
                video_labels.append(label)
                n = os.listdir(folder)
                # n = [img for img in n if '.jpg' in img]
                # num_frames.append(len(n))
                if self.path_annotations:
                    ann_file = os.path.join(self.path_annotations, cl, v_name+'.json')
                    assert os.path.isfile(ann_file), "Annotation file:{} does not exist!!!".format(ann_file)
                    annotations.append(ann_file)

        return video_names, video_labels, annotations
    
    def __call__(self):
        data = self.load_annotation_data()
        split = self.split()
        paths, labels, annotations = self.get_video_names_and_labels(data, split)
        return paths, labels, annotations


        
CATEGORY_ALL = 2
CATEGORY_POS = 1
CATEGORY_NEG = 0

class MakeRWF2000():
    def __init__(self, 
                root,
                train,
                category=CATEGORY_ALL,
                path_annotations=None, 
                path_feat_annotations=None,
                path_person_detections=None,
                shuffle=False):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.path_feat_annotations = path_feat_annotations
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["NonFight", "Fight"]
        self.category = category
        self.shuffle = shuffle
    
    def classes(self):
        return self.classes
    
    def split(self):
        split = "train" if self.train else "val"
        return split
    
    def all_categories(self, split):
        paths = []
        labels = []
        annotations = []
        feat_annotations = []
        for idx, cl in enumerate(self.classes):
            for video_sample in os.scandir(os.path.join(self.root, split, cl)):
                if video_sample.is_dir():
                    paths.append(os.path.join(self.root, split, cl, video_sample))
                    labels.append(idx)
                    if self.path_annotations:
                        assert os.path.exists(os.path.join(self.path_annotations, split, cl, video_sample.name +'.json')), "Annotation does not exist!!!"
                        annotations.append(os.path.join(self.path_annotations, split, cl, video_sample.name +'.json'))
                    if self.path_feat_annotations:
                        assert os.path.exists(os.path.join(self.path_feat_annotations, split, cl, video_sample.name +'.txt')), "Feature annotation does not exist!!!"
                        feat_annotations.append(os.path.join(self.path_feat_annotations, split, cl, video_sample.name +'.txt'))
        
        return paths, labels, annotations
    
    def positive_category(self, split):
        paths = []
        labels = []
        annotations = []
        feat_annotations = []
        label = 1
        label_name = self.classes[label]
        for video_sample in os.scandir(os.path.join(self.root, split, label_name)):
            if video_sample.is_dir():
                paths.append(os.path.join(self.root, split, label_name, video_sample))
                labels.append(label)
                if self.path_annotations:
                    assert os.path.exists(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json')), "Annotation does not exist!!!"
                    annotations.append(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json'))
                if self.path_feat_annotations:
                    assert os.path.exists(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt')), "Feature annotation does not exist!!!"
                    feat_annotations.append(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt'))
        
        return paths, labels, annotations
    
    def negative_category(self, split):
        paths = []
        labels = []
        annotations = []
        feat_annotations = []
        label = 0
        label_name = self.classes[label]
        for video_sample in os.scandir(os.path.join(self.root, split, label_name)):
            if video_sample.is_dir():
                paths.append(os.path.join(self.root, split, label_name, video_sample))
                labels.append(label)
                if self.path_annotations:
                    assert os.path.exists(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json')), "Annotation does not exist!!!"
                    annotations.append(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json'))
                if self.path_feat_annotations:
                    assert os.path.exists(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt')), "Feature annotation does not exist!!!"
                    feat_annotations.append(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt'))
        
        return paths, labels, annotations
    
    def __call__(self):
        split = self.split()
        if self.category == CATEGORY_ALL:
            paths, labels, annotations =  self.all_categories(split)
        elif self.category == CATEGORY_POS:
            paths, labels, annotations = self.positive_category(split)
        elif self.category == CATEGORY_NEG:
            paths, labels, annotations = self.negative_category(split)
        
        if self.shuffle:
            c = list(zip(paths, labels, annotations))
            random.shuffle(c)
            paths, labels, annotations = zip(*c)
        
        return paths, labels, annotations
        

from collections import Counter
import random
def JSON_2_tube(json_file):
    """
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            for i, box in  enumerate(f['boxes']):
                f['boxes'][i] = np.asarray(f['boxes'][i])
        # print(decodedArray[0])
        decodedArray = sorted(decodedArray, key = lambda i: i['id'])
        return decodedArray

def _avg_num_tubes(annotations):
    video_num_tubes=[]
    num_tubes=[]
    tube_lengths = []
    for ann in annotations:
        tubes = JSON_2_tube(ann)
        video_num_tubes.append((ann, len(tubes)))
        num_tubes.append(len(tubes))
        for tube in tubes:
            # print('tube[len]:', tube['len'], len(tube['boxes']), len(tube['foundAt']))
            l = 16 if tube['len']>16 else tube['len']
            tube_lengths.append(tube['len'])
    
    def Average(lst):
        return sum(lst) / len(lst)
    
    print('Avg num_tubes: ', Average(num_tubes))
    print('Avg len_tubes: ', Average(tube_lengths))

def _get_num_tubes(annotations, make_func):
    video_num_tubes=[]
    num_tubes=[]
    for ann in annotations:
        tubes = JSON_2_tube(ann)
        video_num_tubes.append((ann, len(tubes)))
        num_tubes.append(len(tubes))
    with open('hockey_num_tubes_{}.txt'.format('train' if make_func.train else 'val'), 'w') as filehandle:
        filehandle.writelines("{},{}\n".format(t[0], t[1]) for t in video_num_tubes)
    
   
    
if __name__=="__main__":
    make_func = MakeRWF2000(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                    train=True,
                    path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/rwf',
                    category=2)
    paths, labels, annotations = make_func()
    print("paths: ", len(paths))
    print("labels: ",len(labels))
    print("annotations: ",len(annotations))

    _avg_num_tubes(annotations)

    # print("no tubes in: ")
    # without_tube=[]
    # for ann in annotations:
    #     tubes = JSON_2_tube(ann)
    #     if len(tubes)==0:
    #         # print(len(tubes))
    #         without_tube.append(ann)
    
    # with open('3without_tube_{}.txt'.format('train' if make_func.train else 'val'), 'w') as filehandle:
    #     filehandle.writelines("%s\n" % t for t in without_tube)

    # tubes = JSON_2_tube('/media/david/datos/Violence DATA/ActionTubes/RWF-2000/train/Fight/C8wt47cphU8_0.json')
    # print("tubes: ",len(tubes))

    
    ###################################################################################################################################
    # make_func = MakeHockeyDataset(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/HockeyFightsDATASET/frames', 
    #                 train=False,
    #                 cv_split_annotation_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/hockey_jpg1.json',
    #                 path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/hockey')
    # paths, labels, annotations = make_func()
    # print("paths: ", len(paths))
    # print("labels: ", len(labels))
    # print("annotations: ", len(annotations))

    # _avg_num_tubes(annotations)
    # _get_num_tubes(annotations, make_func)
    ###################################################################################################################################

    # m = MakeUCFCrime2Local(root='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
    #                         annotation_path='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/readme',
    #                         bbox_path='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/readme/Txt annotations',
    #                         train=False)
    # paths, labels, annotations, intervals = m()
    # idx=22
    # print(paths[idx])
    # print(labels[idx])
    # print(annotations[idx][0:10])
    # print(intervals[idx])

    # m = MakeUCFCrime2LocalClips(root_anomaly='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips/anomaly',
    #                             root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
    #                             path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos')
    # paths, labels, annotations = m()
    # # idx= random.randint(0, len(paths)-1)
    # idx=65
    # print(idx)
    # print(Counter(labels))
    # print(paths[idx])
    # print(labels[idx])
    # print(annotations[idx])

    # anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    # m.plot(paths[idx], anns)

    ###################################################################################################################################
    # make_func = MakeRLVDDataset(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RealLifeViolenceDataset/frames', 
    #                 train=False,
    #                 cv_split_annotation_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/RealLifeViolenceDataset1.json',
    #                 path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/RealLifeViolenceDataset')
    # paths, labels, annotations, num_frames = make_func()
    # print("paths: ", len(paths))
    # print("labels: ", len(labels))
    # print("annotations: ", len(annotations))
    # print("num_frames: ", len(num_frames))
    # _avg_num_tubes(annotations)

    # print(paths[33:40])
    # print(labels[33:40])
    # print(annotations[33:40])
    # print(num_frames[33:40])
