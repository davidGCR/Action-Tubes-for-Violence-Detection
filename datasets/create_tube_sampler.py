from datasets.tube_crop import TubeCrop
from transformations.temporal_transforms import CenterCrop, RandomCrop

def get_sampler(cfg, train_set):
    """Load sampler: TubeCrop, RandomCrop, CenterCrop

    Args:
        cfg (yaml): cfg.TUBE_DATASET
        train_set (bool): Flac to use RanmdomCrop in train set and CenterCrop in val set.

    Returns:
        sampler: Function to sample frames.
    """
    if cfg.USE_TUBES:
        sampler = TubeCrop(tube_len=cfg.NUM_FRAMES,
                                central_frame=True,
                                max_num_tubes=cfg.NUM_TUBES,
                                # input_type=transf_config['input_1'].itype,
                                sample_strategy=cfg.FRAMES_STRATEGY,
                                random=cfg.RANDOM,
                                box_as_tensor=False)
    else:
        if train_set:
            sampler = RandomCrop(size=cfg.NUM_FRAMES,
                                        stride=1,
                                        input_type='rgb')
        else:
            sampler = CenterCrop(size=cfg.NUM_FRAMES,
                                        stride=1,
                                        input_type='rgb')
    return sampler