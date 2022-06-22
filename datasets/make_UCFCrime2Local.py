import os
import numpy as np
import re
import cv2

class MakeUCFCrime2Local():
    def __init__(self, root, annotation_path, bbox_path, train):
        self.root = root
        self.annotation_path = annotation_path
        self.bbox_path = bbox_path
        self.train = train
 
    def split(self):
        split = "Train_split_AD.txt" if self.train else "Test_split_AD.txt"
        return split
    
    def sp_annotation(self, path):
        """
        1   Track ID. All rows with the same ID belong to the same path.
        2   xmin. The top left x-coordinate of the bounding box.
        3   ymin. The top left y-coordinate of the bounding box.
        4   xmax. The bottom right x-coordinate of the bounding box.
        5   ymax. The bottom right y-coordinate of the bounding box.
        6   frame. The frame that this annotation represents.
        7   lost. If 1, the annotation is outside of the view screen.
        8   occluded. If 1, the annotation is occluded.
        9   generated. If 1, the annotation was automatically interpolated.
        10  label. The label for this annotation, enclosed in quotation marks.
        11+ attributes. Each column after this is an attribute.
        """
        assert os.path.isfile(path), "Txt Annotation {} Not Found!!!".format(path)

        annotations = []
        with open(path) as fid:
            lines = fid.readlines()
            ss = 1 if lines[0].split()[5] == '0' else 0
            for line in lines:
                # v_name = line.split()[0]
                # print(line.split())
                ann = line.split()
                frame_number = int(ann[5]) + ss
                valid = ann[6]
                if valid == '0':
                    annotations.append(
                        {
                            "frame": frame_number,
                            "xmin": ann[1],
                            "ymin": ann[2],
                            "xmax": ann[3],
                            "ymax": ann[4]
                        }
                    )
        positive_intervals = self.positive_segments(annotations)
        return annotations, positive_intervals
                    
    def positive_segments(self, annotations):
        frames = []
        positive_intervals = []
        for an in annotations:
            frames.append(int(an["frame"]))
        frames.sort()
        start_end = np.diff((np.diff(frames) == 1) + 0, prepend=0, append=0)
        # Look for where it flips from 1 to 0, or 0 to 1.
        start_idx = np.where(start_end == 1)[0]
        end_idx = np.where(start_end == -1)[0]

        # print("---- start_idx", start_idx)
        # print("---- end_idx", end_idx, end_idx.shape)
        for s, e in zip(start_idx,end_idx):
            # print("---- ", s,e)
            # print("[{},{}]".format(frames[s], frames[e]))
            positive_intervals.append((frames[s], frames[e]))
        
        return positive_intervals

    def __call__(self):
        split_file = os.path.join(self.annotation_path, self.split())
        paths = []
        labels = []
        annotations = []
        positive_intervals = []
        with open(split_file) as fid:
            lines = fid.readlines()
            for line in lines:
                v_name = line.split()[0]
                # print(v_name[0])
                if os.path.isdir(os.path.join(self.root, v_name)):
                    paths.append(os.path.join(self.root, v_name))
                    label = 0 if "Normal" in v_name else 1
                    labels.append(label)
                    if label==1:
                        annotation, intervals = self.sp_annotation(os.path.join(self.bbox_path, v_name+".txt"))
                        annotations.append(annotation)
                        positive_intervals.append(intervals)
                    else:
                        annotations.append(None)
                        positive_intervals.append(None)
                else:
                    print("Folder ({}) not found!!!".format(v_name))
        
        return paths, labels, annotations, positive_intervals


class MakeUCFCrime2LocalClips():
    def __init__(self, root, path_annotations, path_person_detections, abnormal):
        self.root = root
        self.path_annotations = path_annotations
        self.path_person_detections = path_person_detections
        self.classes = ['normal', 'anomaly'] #Robbery,Stealing
        self.subclasses = ['Arrest', 'Assault'] #Robbery,Stealing
        self.abnormal = abnormal
    
    def __get_list__(self, path):
        paths = os.listdir(path)
        paths = [os.path.join(path,pt) for pt in paths if os.path.isdir(os.path.join(path,pt))]
        return paths
    
    def __annotation__(self, folder_path):
        v_name = os.path.split(folder_path)[1]
        annotation = [ann_file for ann_file in os.listdir(self.path_annotations) if ann_file.split('.')[0] in v_name.split('(')]
        annotation = annotation[0]
        # print('annotation: ',annotation)
        return os.path.join(self.path_annotations, annotation)

    # def __annotation_p_detections__(self, folder_path):

    
    def ground_truth_boxes(self, video_folder, ann_path):
        frames = os.listdir(video_folder)
        frames_numbers = [int(re.findall(r'\d+', f)[0]) for f in frames]
        frames_numbers.sort()
        # print(frames_numbers)

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
        # tmp = [a['frame'] for a in annotations]
        # print(tmp)
        
        return annotations
    
    def plot(self, folder_imgs, annotations_dict, live_paths=[]):
        imgs = os.listdir(folder_imgs)
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        
        imgs.sort(key=natural_keys)
        # print(type(folder_imgs),type(f_paths[0]))
        f_paths = [os.path.join(folder_imgs, ff) for ff in imgs]
        
        for img_path in f_paths:
            print(img_path)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            f_num = os.path.split(img_path)[1]
            f_num = int(re.findall(r'\d+', f_num)[0])
            ann = [ann for ann in annotations_dict if ann['frame']==f_num][0]
            x1 = ann["xmin"]
            y1 = ann["ymin"]
            x2 = ann["xmax"]
            y2 = ann["ymax"]
            cv2.rectangle(image,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0,238,238),
                            1)
            if len(live_paths)>0:
                frame = img_path.split('/')[-1]
                
                for l in range(len(live_paths)):
                    
                    foundAt = True if frame in live_paths[l]['frames_name'] else False
                    if foundAt:
                        idx = live_paths[l]['frames_name'].index(frame)
                        bbox = live_paths[l]['boxes'][idx]
                        x1 = bbox[0]
                        y1 = bbox[1]
                        x2 = bbox[2]
                        y2 = bbox[3]
                        cv2.rectangle(image,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (255,0,0),
                                    1)
            cv2.namedWindow('FRAME'+str(f_num),cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FRAME'+str(f_num), (600,600))
            image = cv2.resize(image, (600,600))
            cv2.imshow('FRAME'+str(f_num), image)
            key = cv2.waitKey(250)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()

    def __call__(self):
        root_anomaly = os.path.join(self.root, self.classes[1])
        root_normal = os.path.join(self.root, self.classes[0])

        # root_anomaly_p_detections = os.path.join(self.path_person_detections, self.classes[1])
        
        if self.abnormal:
            abnormal_paths = self.__get_list__(root_anomaly)
            paths = abnormal_paths
            annotations_anomaly = [self.__annotation__(pt) for pt in abnormal_paths]
            annotations = annotations_anomaly
            labels = [1]*len(abnormal_paths)
            annotations_p_detections = []
            num_frames = []
            for ap in abnormal_paths:
                assert os.path.isdir(ap), 'Folder does not exist!!!'
                n = len(os.listdir(ap))
                num_frames.append(n)
                sp = ap.split('/')
                p_path = os.path.join(self.path_person_detections, sp[-2], sp[-1]+'.json')
                assert os.path.isfile(p_path), 'P_annotation does not exist!!!'
                annotations_p_detections.append(p_path)

        else:
            normal_paths = self.__get_list__(root_normal)
            normal_paths = [path for path in normal_paths if "Normal" in path]
            paths = normal_paths
            annotations_normal = [None]*len(normal_paths)
            annotations = annotations_normal
            labels = [0]*len(normal_paths)
            annotations_p_detections = [None]*len(normal_paths)
            num_frames = []
            for ap in normal_paths:
                assert os.path.isdir(ap), 'Folder does not exist!!!'
                n = len(os.listdir(ap))
                num_frames.append(n)
        # paths = abnormal_paths + normal_paths
        # annotations = annotations_anomaly + annotations_normal
        # labels = [1]*len(abnormal_paths) + [0]*len(normal_paths)
        
        return paths, labels, annotations, annotations_p_detections, num_frames