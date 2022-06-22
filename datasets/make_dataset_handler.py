from utils.global_var import *
from utils.dataset_utils import read_JSON_ann
from datasets.make_dataset import MakeRWF2000, MakeHockeyDataset, MakeRLVDDataset
from datasets.make_UCFCrime import MakeUCFCrime
from datasets.make_UCFCrime2Local import MakeUCFCrime2LocalClips
from datasets.make_cctvfights import make_CCTVFights_dataset_val, make_CCTVFights_dataset_clips

def load_make_dataset(cfg,
                      env_datasets_root,
                      min_clip_len=0,
                      train=True,
                      category=2, 
                      shuffle=False):
    """[summary]

    Args:
        cfg (yaml): cfg.DATA
        env_datasets_root (str): Path to datasets folder    
        min_clip_len (int, optional): Filter videos with min number of frames, only for CCTVFights. Use for CCTVFights dataset. Defaults to 0.
        train (bool, optional): [description]. Defaults to True.
        category (int, optional): Only for RWF-2000. Defaults to 2.
        shuffle (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    home_path =     env_datasets_root
    at_path =       cfg.ACTION_TUBES_FOLDER
    dataset_name =  cfg.DATASET
    cv_split =      cfg.CV_SPLIT
    load_gt =       cfg.LOAD_GROUND_TRUTH

    # print('home_path: ', home_path)
    if dataset_name == RWF_DATASET:
        make_dataset = MakeRWF2000(
            root=os.path.join(home_path, 'RWF-2000/frames'),
            train=train,
            category=category, 
            # path_annotations=os.path.join(home_path, at_path, 'final/rwf'),
            path_annotations=os.path.join(home_path, at_path, 'RWF-2000'),
            shuffle=shuffle)

    elif dataset_name == HOCKEY_DATASET:
        make_dataset = MakeHockeyDataset(
            root=os.path.join(home_path, 'HockeyFightsDATASET/frames'), 
            train=train,
            cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/hockey_jpg{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
            path_annotations=os.path.join(home_path, at_path, 'HockeyFightsDATASET'),
            )
    elif dataset_name == RLVSD_DATASET:
        make_dataset = MakeRLVDDataset(
            root=os.path.join(home_path, 'RealLifeViolenceDataset/frames'), 
            train=train,
            cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/RealLifeViolenceDataset{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
            path_annotations=os.path.join(home_path, at_path, 'RealLifeViolenceDataset'),
            )
    # elif dataset_name == UCFCrime_DATASET:
    #     ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl') if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
    #     make_dataset = MakeUCFCrime(
    #         root=os.path.join(home_path, 'UCFCrime/frames'), 
    #         sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
    #         sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
    #         action_tubes_path=os.path.join(home_path,'ActionTubes/UCFCrime_reduced', ann_file[1]),
    #         train=train,
    #         ground_truth_tubes=False)
    elif dataset_name == UCFCrimeReduced_DATASET:
        ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl') if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
        make_dataset = MakeUCFCrime(
            root=os.path.join(home_path, 'UCFCrime_Reduced', 'frames'), 
            sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
            sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
            action_tubes_path=os.path.join(home_path, at_path, 'UCFCrime_Reduced'),
            train=train,
            ground_truth_tubes=load_gt)
    elif dataset_name == CCTVFight_DATASET and train:
        root = os.path.join(home_path, "CCTVFights/frames/fights")
        json_file = os.path.join(home_path, "CCTVFights/groundtruth_modified.json")
        pers_annotations_folder = os.path.join(home_path, "PersonDetections/CCTVFights/fights")
        make_dataset = make_CCTVFights_dataset_clips(root, json_file, pers_annotations_folder, min_clip_len, "training")
    elif dataset_name == CCTVFight_DATASET and not train:
        root = Path(home_path)
        dataset_root = root / "CCTVFights/frames/fights"
        root_person_detec = root / "PersonDetections/CCTVFights"
        json_file = root / "CCTVFights/groundtruth_modified.json"
        make_dataset = make_CCTVFights_dataset_val(dataset_root, root_person_detec, json_file, "testing")
    else:
        print('Invalid DATASET name!!!')
        exit()

    return make_dataset

def load_make_dataset_UCFCrime2Local(data_root):
    """Load MakeUCFCrime2LocalClips make function

    Args:
        data_root (pathlib.Path): Root path to folder with datasets
    """
    val_make_dataset = MakeUCFCrime2LocalClips(
                root=data_root/'UCFCrime2Local/UCFCrime2LocalClips',
                # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
                path_annotations=data_root/'UCFCrime2Local/Txt annotations-longVideos',
                path_person_detections=data_root/'PersonDetections/ucfcrime2local',
                abnormal=True)
    return val_make_dataset