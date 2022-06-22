from torchvision import transforms
from datasets.make_UCFCrime2Local import MakeUCFCrime2LocalClips
from datasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
from tubes.run_tube_gen import extract_tubes_from_video
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from utils.utils import natural_sort
from utils.tube_utils import JSON_2_videoDetections
import os

def debug_load_ucfcrime2local_dataset(data_root):
    """Test MakeUCFCrime2LocalClips make function

    Args:
        data_root (str): Root path to folder with datasets
    """
    val_make_dataset = MakeUCFCrime2LocalClips(
                root=data_root/'UCFCrime2Local/UCFCrime2LocalClips',
                # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
                path_annotations=data_root/'UCFCrime2Local/Txt annotations-longVideos',
                path_person_detections=data_root/'PersonDetections/ucfcrime2local',
                abnormal=True)
    paths, labels, annotations, annotations_p_detections, num_frames = val_make_dataset()
    print("paths: \n\t-", len(paths), "\n\t-", paths[0:3])
    print("labels: \n\t-", len(labels), "\n\t-", labels[0:3])
    print("annotations: \n\t-", len(annotations), "\n\t-", annotations[0:3])
    print("annotations_p_detections: \n\t-", len(annotations_p_detections), "\n\t-", annotations_p_detections[0:3])
    print("num_frames: \n\t-", len(num_frames), "\n\t-", num_frames[0:3])

def debug_ucfcrime2localclips_dataset(val_make_dataset, transformations):
    paths, labels, annotations, annotations_p_detections, num_frames = val_make_dataset()
    for i, (path, label, annotation, annotation_p_detections, n_frames) in enumerate(zip(paths, labels, annotations, annotations_p_detections, num_frames)):
        print('{}--video:{}, num_frames: {}'.format(i+1, path, n_frames))
        print('----annotation:{}, p_detec: {}, {}'.format(annotation, annotation_p_detections, type(annotation_p_detections)))
        video_dataset = UCFCrime2LocalVideoDataset(
            path=path,
            sp_annotation=annotation,
            transform=transforms.ToTensor(),
            clip_len=n_frames, #all frames
            clip_temporal_stride=1,
            transformations=transformations
        )
        frames_names = os.listdir(path)
        frames_names = natural_sort(frames_names)
        frames_indices = list(range(0,len(frames_names)))
        person_detections = JSON_2_videoDetections(annotation_p_detections)
        TUBE_BUILD_CONFIG['dataset_root'] = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips'
        TUBE_BUILD_CONFIG['person_detections'] = person_detections

        for clip, frames_name_, gt, num_frames in video_dataset:
            print("clip: \n\t-", len(clip), "\n\t-", clip[0:3])
            print("frames_name_: \n\t-", len(frames_name_), "\n\t-", frames_name_)
            print("gt: \n\t-", len(gt), "\n\t-", gt[0:3])
            print("num_frames: \n\t-", num_frames)
            
            
            live_paths, time = extract_tubes_from_video(frames_indices, frames_names, MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG)
            print('live_paths: ', len(live_paths))
        exit()