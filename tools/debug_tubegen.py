from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset_utils import read_JSON_ann
from datasets.CCTVFights_dataset import *
from datasets.make_cctvfights import  make_CCTVFights_dataset_val
from datasets.collate_fn import my_collate
from tubes.run_tube_gen import *
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from utils.tube_utils import JSON_2_videoDetections
from pathlib import Path


def test_tubegen_CCTVFights_dataset(cfg, transforms):
    """Test Sequential Dataset to load long videos clip by clip. 
        Extract action tubes from each sequence and save as json
    """
    # root = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CCTVFights/frames/fights"
    root = Path("/media/david/datos/Violence DATA")
    dataset_root = root / "CCTVFights/frames/fights"
    root_person_detec = root / "PersonDetections/CCTVFights"
    json_file = root / "CCTVFights/groundtruth_modified.json"
    data = read_JSON_ann(json_file)
    tubes_path = root / "ActionTubesV2/CCTVFights/test/fights"
    print(data["version"])

    paths, frame_rates, tmp_annotations, person_det_files = make_CCTVFights_dataset_val(dataset_root, 
                                                                                        root_person_detec, 
                                                                                        json_file,
                                                                                        "testing")()
    # transform = transforms.ToTensor()

    for j, (path, frame_rate, tmp_annot, pers_detect_annot) in enumerate(zip(paths, frame_rates, tmp_annotations, person_det_files)):
        print(j, path)
        print("tmp_annot: ", tmp_annot)
        print("pers_detect_annot: ", pers_detect_annot)
        print("frame_rate: ", frame_rate)
        dataset = SequentialDataset(cfg=cfg,
                                    seq_len=32, 
                                    tubes_path=tubes_path, 
                                    pers_detect_annot=pers_detect_annot, 
                                    annotations=tmp_annot, 
                                    video_path=path, 
                                    frame_rate=frame_rate, 
                                    transforms=transforms)

        loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        collate_fn=my_collate
                        )
        # for video_boxes, video_images, label, tubes, keyframes in loader:
        for video_boxes, video_images, label, path, keyframes in loader:
            # frames_names = [list(i) for i in zip(*frames_names)]
            print('\tprocessing clip: VIDEO_IMGS: {}, KEYFRAMES: {}, BOXES: {}, LABEL: {}'.format(video_images.size(),
                                                                                    keyframes.size(), 
                                                                                    video_boxes.size(),
                                                                                    label))