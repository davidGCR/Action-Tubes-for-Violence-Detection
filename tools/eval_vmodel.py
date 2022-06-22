import add_libs
from configs.defaults import get_cfg_defaults
from utils.global_var import *
from lib.optimization import val, val_map, validate_long_videos
from lib.optimization_mil import val_regressor, val_regressor_UCFCrime2Local
from lib.accuracy import calculate_accuracy_2, calculate_accuracy_regressor

from utils.utils import get_torch_device, load_checkpoint, count_parameters
from datasets.make_dataset_handler import load_make_dataset, load_make_dataset_UCFCrime2Local
from datasets.dataloaders import data_with_tubes, data_with_tubes_localization, data_with_tubes_val
from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam
import torch

def main(h_path):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3DRoiPool_2DRoiPool.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path

    device = get_torch_device()
    
    min_clip_len = cfg.TUBE_DATASET.STRIDE*cfg.TUBE_DATASET.SEQ_LEN if cfg.DATA.DATASET == CCTVFight_DATASET else 0
    if cfg.MODEL._HEAD.NAME == BINARY:
        make_dataset_val = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=min_clip_len,
                                        train=False,
                                        category=2,
                                        shuffle=False)                           
        val_loader, val_dataset, transforms_val = data_with_tubes_val(cfg, make_dataset_val)
    
    elif cfg.MODEL._HEAD.NAME == REGRESSION:
        if cfg.DATA.DATASET == UCFCrimeReduced_DATASET:
            make_dataset_val = load_make_dataset_UCFCrime2Local(Path(cfg.ENVIRONMENT.DATASETS_ROOT))
        elif cfg.DATA.DATASET == RWF_DATASET:
            make_dataset_val = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=min_clip_len,
                                        train=False,
                                        category=2,
                                        shuffle=False)                           
            train_loader, val_loader, train_dataset, val_dataset, transforms_train, transforms_val = data_with_tubes(cfg, None, make_dataset_val)
    else:
        print("Error: Unrecognized head name!!!")
        raise NotImplementedError()

    model = TwoStreamVD_Binary_CFam(cfg.MODEL).to(device)
    params_num = count_parameters(model)
    print("Num parameters: ", params_num)

    model, _, _, _, _ = load_checkpoint(model, device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)

    for epoch in range(0, cfg.MODEL.INFERENCE.REPETITIONS):
        if cfg.MODEL._HEAD.NAME == BINARY:
            if not cfg.DATA.DATASET == CCTVFight_DATASET:
                val_loss, val_acc = val(
                    val_loader,
                    epoch, 
                    model, 
                    None,
                    device,
                    cfg.TUBE_DATASET.NUM_TUBES,
                    calculate_accuracy_2)
            else:
                mAP = validate_long_videos(cfg.TUBE_DATASET, 
                                     make_dataset_val, 
                                     32, 
                                    #  os.path.join(cfg.ENVIRONMENT.DATASETS_ROOT, "ActionTubesV2/CCTVFights/test/fights"),
                                     os.path.join(cfg.TUBE_DATASET.TUBE_FOLDER, "test/fights"),
                                     transforms_val,
                                     epoch,
                                     epoch=epoch,
                                     model=model,
                                     criterion=None,
                                     device=device,
                                     num_tubes=cfg.TUBE_DATASET.NUM_TUBES)
        elif cfg.MODEL._HEAD.NAME == REGRESSION:        
            if cfg.DATA.DATASET == UCFCrimeReduced_DATASET:
                if (epoch+1)%cfg.SOLVER.VALIDATE_EVERY == 0:
                    ap05, ap02 = val_regressor_UCFCrime2Local(cfg.TUBE_DATASET,
                                            make_dataset_val, 
                                            TWO_STREAM_INPUT_val, 
                                            model, 
                                            device, 
                                            epoch,
                                            Path(cfg.ENVIRONMENT.DATASETS_ROOT)/"UCFCrime2Local/UCFCrime2LocalClips",
                                            Path(cfg.ENVIRONMENT.DATASETS_ROOT)/"ActionTubesV2/UCFCrime2LocalClips")
            elif cfg.DATA.DATASET == RWF_DATASET:
                val_loss, val_acc = val_regressor(
                    val_loader,
                    epoch, 
                    model, 
                    None,
                    device,
                    cfg.TUBE_DATASET.NUM_TUBES,
                    calculate_accuracy_regressor)
        else:
            print("Error: Unrecognized head name!!!")
            raise NotImplementedError()

if __name__=='__main__':
    h_path = HOME_COLAB
    torch.autograd.set_detect_anomaly(True)
    main(h_path)