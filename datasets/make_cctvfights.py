from utils.dataset_utils import read_JSON_ann
import os
import numpy as np

class make_CCTVFights_dataset_clips():
    def __init__(self, root, json_file, pers_annotations_folder, min_clip_len, subset='training'):
        """Load clip paths considering each positive and negative instance as an individual sample

        Args:
            root (str): Path of the dataset folder.
            json_file (str): Annotations ground truth path.
            subset (str, optional): Subset to load. Defaults to 'training'.
            min_clip_len (int): filter clips with min number of frames.

        Returns:
            (list, list, list, list, list) : Lists of paths, labels, indices, tmp_annotations = [{'segment': [142.2, 230.68], 'label': 'NonFight', 'frame_rate': 25.0}, ...], pers_annotation
        """
        self.data = read_JSON_ann(json_file)
        self.root = root
        self.pers_annotations_folder = pers_annotations_folder
        self.subset = subset
        self.min_clip_len = min_clip_len
    
    def getClip(self, tmp_annotation):
        frame_rate = tmp_annotation["frame_rate"]
        clip_start = round(tmp_annotation["segment"][0]*frame_rate)
        clip_start = 1 if clip_start == 0 else clip_start
        clip_end = round(tmp_annotation["segment"][1]*frame_rate)
        
        # clip = list(range(clip_start, clip_end+1, 1))
        clip = np.arange(clip_start, clip_end+1).tolist()
        return clip
    
    def __call__(self):
        paths = []
        labels = []
        indices = []
        tmp_annotations = []
        pers_annotations = []
        clips = []
        for i, key in enumerate(self.data["database"]):
            if self.data["database"][key]["subset"] == self.subset:
                for j in range(len(self.data["database"][key]["annotations"])):
                    temp_annot = self.data["database"][key]["annotations"][j]
                    temp_annot["frame_rate"] = self.data["database"][key]["frame_rate"]
                    clip = self.getClip(temp_annot)
                    if len(clip)>=self.min_clip_len:
                        paths.append(os.path.join(self.root, key))
                        labels.append(1)
                        indices.append(j)
                        tmp_annotations.append(temp_annot)
                        pers_annotations.append(os.path.join(self.pers_annotations_folder, key+'.json'))
                        clips.append(clip)
                for k in range(len(self.data["database"][key]["n_annotations"])):
                    temp_annot = self.data["database"][key]["n_annotations"][k]
                    temp_annot["frame_rate"] = self.data["database"][key]["frame_rate"]
                    clip = self.getClip(temp_annot)
                    if len(clip)>=self.min_clip_len:
                        paths.append(os.path.join(self.root, key))
                        labels.append(0)
                        indices.append(k)
                        tmp_annotations.append(temp_annot)
                        pers_annotations.append(os.path.join(self.pers_annotations_folder, key+'.json'))
                        clips.append(clip)
        return paths, labels, indices, tmp_annotations, pers_annotations, clips

class make_CCTVFights_dataset_val():
    def __init__(self, root, root_person_detec, json_file, subset='testing'):
        """Load clip paths considering long videos.

        Args:
            root (str): Path of the dataset folder.
            root_person_detec (str): Path of the person detections folder.
            json_file (str): Annotations ground truth path.
            subset (str, optional): Subset to load. Defaults to 'testing'.

        Returns:
            (list, list, list, list) : Lists of paths, frame_rates, tmp_annotations, person_det_files
        """
        self.data = read_JSON_ann(json_file)
        self.root = root
        self.root_person_detec = root_person_detec
        self.subset = subset
    
    def __call__(self):
        paths = []
        frame_rates = []
        tmp_annotations = []
        person_det_files = []
        for i, key in enumerate(self.data["database"]):
            if self.data["database"][key]["subset"] == self.subset:
                paths.append(os.path.join(self.root, key))
                frame_rates.append(self.data["database"][key]["frame_rate"])
                tmp_annotations.append(self.data["database"][key]["annotations"])
                person_det_files.append(os.path.join(self.root_person_detec, 'fights', key+'.json'))
        return paths, frame_rates, tmp_annotations, person_det_files