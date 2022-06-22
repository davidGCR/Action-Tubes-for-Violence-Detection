from utils.global_var import *

from datasets.tube_dataset import TubeDataset
from datasets.collate_fn import my_collate
from datasets.cnn_input_config import CnnInputConfig
from datasets.CCTVFights_dataset import ClipDataset, SequentialDataset
from datasets.dynamicImage_dataset import DynamicImageDataset
from transformations.model_transforms import *
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

def get_sampler(labels):
    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1./class_sample_count
    print('class_sample_count: ', class_sample_count)
    print('weight: ', weight)
    samples_weight = weight[labels]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def two_stream_transforms(keyframe_strategy):
    """Spatial transformations for two stream model

    Args:
        keyframe_strategy (int): Keyframe strategy

    Returns:
        tuple: transforms_config_train, transforms_config_val
    """
    transforms_config_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(keyframe_strategy, 'train') 
    }
    transforms_config_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None),
        'input_2': load_key_frame_config(keyframe_strategy, 'val') 
    }
    return transforms_config_train, transforms_config_val
    

def load_key_frame_config(keyframe_strategy, split):
    """Build config dict for input of 2d_Branch

    Args:
        keyframe_strategy (int): Keyframe strategy
        split (str): split train or val

    Returns:
        CnnInputConfig: object with config of the 2d branch
    """
    if keyframe_strategy in [RGB_BEGIN_KEYFRAME, RGB_MIDDLE_KEYFRAME, RGB_RANDOM_KEYFRAME]:
        input_2_c = CnnInputConfig()
        input_2_c.itype = RGB_FRAME
        input_2_c.spatial_transform = resnet_transf()[split]

    elif keyframe_strategy == DYNAMIC_IMAGE_KEYFRAME:
        input_2_c = CnnInputConfig()
        input_2_c.itype = DYN_IMAGE
        input_2_c.spatial_transform = resnet_di_transf()[split]
    else:
        print('Error loading key_frame_config. No valid keyframe...')
        exit()
    return input_2_c

def dataloaders_for_di_model(cfg, make_dataset_train, make_dataset_val):
    spatial_transform_train = resnet_di_transf()['train']
    spatial_transform_val = resnet_di_transf()['val']
    train_dataset = DynamicImageDataset(cfg=cfg.DYNAMIC_IMAGE_DATASET, 
                                        make_fn=make_dataset_train, 
                                        train_set=True, 
                                        transform=spatial_transform_train)
    
    val_dataset = DynamicImageDataset(cfg=cfg.DYNAMIC_IMAGE_DATASET, 
                                        make_fn=make_dataset_val, 
                                        train_set=False, 
                                        transform=spatial_transform_val)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.DATALOADER.TRAIN_BATCH,
                              shuffle=True,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              # pin_memory=True,
                              drop_last=cfg.DATALOADER.DROP_LAST)
    val_loader = DataLoader(val_dataset,
                              batch_size=cfg.DATALOADER.VAL_BATCH,
                              shuffle=False,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              # pin_memory=True,
                              drop_last=cfg.DATALOADER.DROP_LAST)
    
    return train_loader, val_loader, train_dataset, val_dataset
    
    
def data_with_tubes(cfg, make_dataset_train, make_dataset_val):
    """Build dataloaders for train and val sets.

    Args:
        cfg (yaml): Main yaml file
        make_dataset_train (function): make function of train set
        make_dataset_val (function): make function of val/test set

    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset) dataloaders and datasets
    """
    transforms_config_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'train') 
    }
    transforms_config_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'val') 
    }

    if cfg.DATA.DATASET == 'CCTVFights':
        train_dataset = ClipDataset(
                            cfg=cfg.TUBE_DATASET,
                            tube_folder=os.path.join(cfg.TUBE_DATASET.TUBE_FOLDER, 'train'),
                            seq_len=cfg.TUBE_DATASET.SEQ_LEN, 
                            stride=cfg.TUBE_DATASET.STRIDE, 
                            make_fn=make_dataset_train, 
                            random=cfg.TUBE_DATASET.RANDOM_INSTANCE_CLIP, 
                            transforms=transforms_config_train,
                            train_set=True
                            )
        train_loader = DataLoader(train_dataset,
                            batch_size=cfg.DATALOADER.TRAIN_BATCH,
                            # shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            # pin_memory=True,
                            collate_fn=my_collate,
                            sampler=get_sampler(train_dataset.labels),
                            drop_last=cfg.DATALOADER.DROP_LAST
                            )
        val_dataset = None
        val_loader = None
    else:
        train_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_train, transforms_config_train, cfg.DATA.DATASET, True)
        train_loader = DataLoader(train_dataset,
                            batch_size=cfg.DATALOADER.TRAIN_BATCH,
                            # shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            # pin_memory=True,
                            collate_fn=my_collate,
                            sampler=get_sampler(train_dataset.labels),
                            drop_last=cfg.DATALOADER.DROP_LAST
                            )
        val_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_val, transforms_config_val, cfg.DATA.DATASET, False)
        val_loader = DataLoader(val_dataset,
                            batch_size=cfg.DATALOADER.VAL_BATCH,
                            # shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            sampler=get_sampler(val_dataset.labels),
                            # pin_memory=True,
                            collate_fn=my_collate,
                            drop_last=cfg.DATALOADER.DROP_LAST
                            )
        time_val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            # shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            sampler=get_sampler(val_dataset.labels),
                            # pin_memory=True,
                            collate_fn=my_collate,
                            drop_last=cfg.DATALOADER.DROP_LAST
                            )
    return train_loader, val_loader, train_dataset, val_dataset, transforms_config_train, transforms_config_val, time_val_loader

def data_with_tubes_val(cfg, make_dataset_val):
    """Build dataloaders for val sets.

    Args:
        cfg (yaml): Main yaml file
        make_dataset_val (function): make function of val/test set

    Returns:
        tuple: (val_loader, val_dataset) dataloaders and datasets
    """
    transforms_config_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'train') 
    }
    transforms_config_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'val') 
    }

    if cfg.DATA.DATASET == 'CCTVFights':
        val_dataset = None
        val_loader = None
    else:
        val_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_val, transforms_config_val, cfg.DATA.DATASET, False)
        val_loader = DataLoader(val_dataset,
                            batch_size=cfg.DATALOADER.VAL_BATCH,
                            # shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            sampler=get_sampler(val_dataset.labels),
                            # pin_memory=True,
                            collate_fn=my_collate,
                            drop_last=cfg.DATALOADER.DROP_LAST
                            )
    return val_loader, val_dataset, transforms_config_val

def dataloaders_for_CCTVFights(cfg, make_dataset_train, make_dataset_val):
    """Build dataloaders for train and val sets specifically for CCTVFights dataset.

    Args:
        cfg (yaml): Main yaml file
        make_dataset_train (function): make function of train set
        make_dataset_val (function): make function of val/test set

    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset) dataloaders and datasets
    """
    transforms_config_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'train') 
    }
    transforms_config_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'val') 
    }

    # train_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_train, TWO_STREAM_INPUT_train, cfg.DATA.DATASET, True)
    # train_loader = DataLoader(train_dataset,
    #                     batch_size=cfg.DATALOADER.TRAIN_BATCH,
    #                     # shuffle=False,
    #                     num_workers=cfg.DATALOADER.NUM_WORKERS,
    #                     # pin_memory=True,
    #                     collate_fn=my_collate,
    #                     sampler=train_dataset.get_sampler(),
    #                     drop_last=cfg.DATALOADER.DROP_LAST
    #                     )
    # val_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_val, TWO_STREAM_INPUT_val, cfg.DATA.DATASET, False)
    # val_loader = DataLoader(val_dataset,
    #                     batch_size=cfg.DATALOADER.VAL_BATCH,
    #                     # shuffle=True,
    #                     num_workers=cfg.DATALOADER.NUM_WORKERS,
    #                     sampler=val_dataset.get_sampler(),
    #                     # pin_memory=True,
    #                     collate_fn=my_collate,
    #                     drop_last=cfg.DATALOADER.DROP_LAST
    #                     )
    # return train_loader, val_loader, train_dataset, val_dataset
    return transforms_config_train, transforms_config_val



def data_with_tubes_localization(cfg, make_dataset_train):
    """Build train dataloader for train dataset and CnnInputConfig for val set.

    Args:
        cfg (yaml): Main yaml file
        make_dataset_train (function): make function of train set
        make_dataset_val (function): make function of val/test set

    Returns:
        tuple: (train_loader, dict{input_1: CnnInputConfig, input_2: CnnInputConfig})
    """

    TWO_STREAM_INPUT_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'train') 
    }

    TWO_STREAM_INPUT_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None), 
        'input_2': load_key_frame_config(cfg.TUBE_DATASET.KEYFRAME_STRATEGY, 'val')
    }
    
    train_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_train, TWO_STREAM_INPUT_train, cfg.DATA.DATASET)
    train_loader = DataLoader(train_dataset,
                        batch_size=cfg.DATALOADER.TRAIN_BATCH,
                        # shuffle=False,
                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                        # pin_memory=True,
                        collate_fn=my_collate,
                        sampler=train_dataset.get_sampler(),
                        drop_last=cfg.DATALOADER.DROP_LAST
                        )
    return train_loader, TWO_STREAM_INPUT_val