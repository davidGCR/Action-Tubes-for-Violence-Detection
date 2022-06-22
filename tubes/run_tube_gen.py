from tubes.incremental_linking import IncrementalLinking
from tubes.motion_segmentation import MotionSegmentation
import time
import torch

def extract_tubes_from_video(frames_indices, frames_names, motion_seg_config, tube_build_config, gt=None):
    """Extract violent action tubes from a video

    Args:
        frames_indices (list): Indices of the frames in the folder.
        frames_names (list): Names of the files/images. They must be in order.
        motion_seg_config (dict): Configuration settings for motion segmentation.
        tube_build_config (dict): Configuration settings for tube building.
        gt (bool, optional): Flac to plot ground truth. Defaults to None.

    Returns:
        tuple: (live_paths, time) List of action tubes and execution time.
    """
    segmentator = MotionSegmentation(motion_seg_config)
    tube_builder = IncrementalLinking(tube_build_config)
    start = time.time()
    live_paths = tube_builder(frames_indices, frames_names, segmentator, gt)
    end = time.time()
    exec_time = end - start
    return  live_paths, exec_time

def extract_tubes_from_sequentialdataset(dataloader, motion_seg_config, tube_build_config, gt=None):
    for video_images, label, frames_names, sequence in dataloader:
        batch_sequence = torch.stack(sequence, dim=0)
        batch_sequence = torch.transpose(batch_sequence, 0, 1)
        frames_names = [list(i) for i in zip(*frames_names)]
        print('\tprocessing clip: NAMES: {}, BATCH_SEQ: {}, VIDEO_IMGS: {}'.format(frames_names, batch_sequence, video_images.size()))
        batch_size = batch_sequence.size()
        for i in range(batch_size[0]):
            f_names = frames_names[i]
            f_indices = batch_sequence[i].cpu().numpy().tolist()

            # print('f_names: ', f_names)

            live_paths, exec_time = extract_tubes_from_video(f_indices, f_names, motion_seg_config, tube_build_config, gt)
            print('live_paths: ', len(live_paths))
            # for k in range(len(live_paths)):
            #     live_paths[k]

