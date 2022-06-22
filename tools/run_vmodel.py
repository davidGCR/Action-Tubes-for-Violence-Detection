from os import error
import add_libs
from configs.defaults import get_cfg_defaults
from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam

from utils.utils import get_torch_device, load_checkpoint, save_checkpoint
from utils.global_var import *
from utils.create_log_name import log_name

from datasets.make_dataset_handler import load_make_dataset, load_make_dataset_UCFCrime2Local
from datasets.dataloaders import data_with_tubes, data_with_tubes_localization

from lib.optimization import train, val, val_map, validate_long_videos, fps
from lib.optimization_mil import train_regressor, val_regressor, val_regressor_UCFCrime2Local
from lib.accuracy import calculate_accuracy_2, calculate_accuracy_regressor

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from tqdm import tqdm
import argparse

def main():
    if args.env == "windows":
        h_path = HOME_WINDOWS
    elif args.env == "ubuntu":
        h_path = HOME_UBUNTU
    elif args.env == "colab":
        h_path = HOME_COLAB
    print('environment,', args.env)
    print('cf,', args.cf)
    print('rt_model,', args.rt_model)
    
    # Setup cfg.
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(WORK_DIR / "configs/ONESTREAM_16RGB_3DRoiPool.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3D_2D_whithoutROILayers.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3DRoiPool_2D_crop.yaml")
    cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3DRoiPool_2DRoiPool.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_3DResNet_16RGB_3DRoiPool_2DRoiPool.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_X3D_16RGB_3DRoiPool_2DRoiPool.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3DRoiPool_2DRoiPool-MIL.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_CCTVFights_16RGB_3DRoiPool_2DRoiPool.yaml")
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_MIL.yaml")
    
    cfg.merge_from_file(WORK_DIR / "configs/{}.yaml".format(args.cf))
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    if args.rt_model is not None:
        cfg.MODEL.RESTORE_TRAIN.ACTIVE = True
        cfg.MODEL.RESTORE_TRAIN.CHECKPOINT_PATH = args.rt_model
    # print(cfg)

    # from debug_model import debug_model, see_models
    # debug_model(cfg.MODEL)
    # see_models()
    # exit()

    # from debug_dataset import test_cctvfights_datasets
    # from datasets.dataloaders import data_with_tubes_for_CCTVFights
    # transforms_config_train, transforms_config_val = data_with_tubes_for_CCTVFights(cfg, None, None)
    # test_cctvfights_datasets(cfg.TUBE_DATASET, transforms_config_train, transforms_config_val)
    # exit()

    device = get_torch_device()
    min_clip_len = cfg.TUBE_DATASET.STRIDE*cfg.TUBE_DATASET.SEQ_LEN if cfg.DATA.DATASET == CCTVFight_DATASET else 0
    if cfg.MODEL._HEAD.NAME == BINARY:
        make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=min_clip_len,
                                        train=True,
                                        category=2,
                                        shuffle=False)
        make_dataset_val = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=min_clip_len,
                                        train=False,
                                        category=2,
                                        shuffle=False)                           
        train_loader, val_loader, train_dataset, val_dataset, transforms_train, transforms_val, time_val_loader = data_with_tubes(cfg, make_dataset_train, make_dataset_val)

        # from debug_tubegen import test_tubegen_CCTVFights_dataset
        # test_tubegen_CCTVFights_dataset(cfg.TUBE_DATASET, transforms_val)
        # exit()
        
        # from debug_dataset import test_tube_dataset
        # test_tube_dataset(train_dataset, val_dataset)
        # exit()
    
    elif cfg.MODEL._HEAD.NAME == REGRESSION:
        make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        train=True,
                                        category=2,
                                        shuffle=False)
        if cfg.DATA.DATASET == UCFCrimeReduced_DATASET:
            make_dataset_val = load_make_dataset_UCFCrime2Local(Path(cfg.ENVIRONMENT.DATASETS_ROOT))
            train_loader, TWO_STREAM_INPUT_val = data_with_tubes_localization(cfg, make_dataset_train)
            # from debug_loc_dataset import debug_ucfcrime2localclips_dataset
            # debug_ucfcrime2localclips_dataset(make_dataset_val, TWO_STREAM_INPUT_val)
        elif cfg.DATA.DATASET == RWF_DATASET:
            make_dataset_val = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=min_clip_len,
                                        train=False,
                                        category=2,
                                        shuffle=False)                           
            train_loader, val_loader, train_dataset, val_dataset, transforms_train, transforms_val, time_val_loader = data_with_tubes(cfg, make_dataset_train, make_dataset_val)

    else:
        print("Error: Unrecognized head name!!!")
        raise NotImplementedError()
    

    model = TwoStreamVD_Binary_CFam(cfg.MODEL).to(device)
    params = model.parameters()
    exp_config_log = log_name(cfg)

    #log
    h_p = HOME_DRIVE if cfg.ENVIRONMENT.DATASETS_ROOT==HOME_COLAB else cfg.ENVIRONMENT.DATASETS_ROOT
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)
    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)
    # print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path_folder)

    if  cfg.SOLVER.OPTIMIZER.NAME == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            params, 
            lr=cfg.SOLVER.LR, 
            eps=1e-8)
    elif cfg.SOLVER.OPTIMIZER.NAME == 'SGD':
        optimizer = torch.optim.SGD(params=params,
                                    lr=cfg.SOLVER.LR,
                                    momentum=0.9,
                                    weight_decay=1e-3)
    elif cfg.SOLVER.OPTIMIZER.NAME == 'Adam':
        optimizer = torch.optim.Adam(
            params, 
            lr=cfg.SOLVER.LR, 
            eps=1e-3, 
            amsgrad=True)            
    
    if cfg.SOLVER.CRITERION == 'CEL':
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.SOLVER.CRITERION == 'BCE':
        criterion = nn.BCELoss().to(device)
    
    start_epoch = 0
    ##Restore training & Transfer learning
    if cfg.MODEL.RESTORE_TRAIN.ACTIVE:
        print('Restoring training from: ', cfg.MODEL.RESTORE_TRAIN.CHECKPOINT_PATH)
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, device, optimizer, cfg.MODEL.RESTORE_TRAIN.CHECKPOINT_PATH)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs
    elif cfg.MODEL.TRANSF_LEARNING.ACTIVE:
        print('TF: Initializing from: ', cfg.MODEL.TRANSF_LEARNING.CHECKPOINT_PATH)
        model, _, _, _, _ = load_checkpoint(model, device, optimizer, cfg.MODEL.TRANSF_LEARNING.CHECKPOINT_PATH)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        verbose=True,
        factor=cfg.SOLVER.OPTIMIZER.FACTOR,
        min_lr=cfg.SOLVER.OPTIMIZER.MIN_LR)
    
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        if cfg.MODEL._HEAD.NAME == BINARY:
            train_loss, train_acc, train_time = train(
                train_loader, 
                epoch,
                cfg.SOLVER.EPOCHS, 
                model, 
                criterion, 
                optimizer, 
                device, 
                cfg.TUBE_DATASET.NUM_TUBES, 
                calculate_accuracy_2)
            writer.add_scalar('training loss', train_loss, epoch)
            writer.add_scalar('training accuracy', train_acc, epoch)
            
            if not cfg.DATA.DATASET == CCTVFight_DATASET:
                val_loss, val_acc = val(
                    val_loader,
                    epoch, 
                    model, 
                    criterion,
                    device,
                    cfg.TUBE_DATASET.NUM_TUBES,
                    calculate_accuracy_2)
                fps(time_val_loader,
                    epoch, 
                    model,
                    device,
                    cfg.TUBE_DATASET.NUM_TUBES)
                scheduler.step(val_loss)
                writer.add_scalar('validation loss', val_loss, epoch)
                writer.add_scalar('validation accuracy', val_acc, epoch)
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
                                     criterion=criterion,
                                     device=device,
                                     num_tubes=cfg.TUBE_DATASET.NUM_TUBES)
                scheduler.step(train_loss)
                writer.add_scalar('mAP', mAP, epoch)
        elif cfg.MODEL._HEAD.NAME == REGRESSION:
            train_loss, train_acc = train_regressor(
                                                    train_loader, 
                                                    epoch, 
                                                    model, 
                                                    criterion, 
                                                    optimizer, 
                                                    device, 
                                                    cfg.TUBE_DATASET.NUM_TUBES, 
                                                    calculate_accuracy_regressor,
                                                    False)

            scheduler.step(train_loss)
            writer.add_scalar('training loss', train_loss, epoch)
            
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
                    writer.add_scalar('AP-0.5', ap05, epoch)
                    writer.add_scalar('AP-0.2', ap02, epoch)
            elif cfg.DATA.DATASET == RWF_DATASET:
                val_loss, val_acc = val_regressor(
                    val_loader,
                    epoch, 
                    model, 
                    criterion,
                    device,
                    cfg.TUBE_DATASET.NUM_TUBES,
                    calculate_accuracy_regressor)
                scheduler.step(val_loss)
                writer.add_scalar('validation loss', val_loss, epoch)
                writer.add_scalar('validation accuracy', val_acc, epoch)

        else:
            print("Error: Unrecognized head name!!!")
            raise NotImplementedError()


        if (epoch+1)%cfg.SOLVER.SAVE_EVERY == 0:
            save_checkpoint(
                model, 
                cfg.SOLVER.EPOCHS, 
                epoch, 
                optimizer,
                train_loss, 
                os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))

if __name__=='__main__':
    # h_path = HOME_WINDOWS
    torch.autograd.set_detect_anomaly(True)
    # files = "\n\n"+"\n -".join([f for f in os.listdir('configs/') if Path(f).suffix == '.yaml'])
    parser = argparse.ArgumentParser(description='Train the TwoStream model')
    parser.add_argument('--env', type=str, required=True, help='enviroment where execute experiments.')
    parser.add_argument('--cf', type=str, required=True)#, help='Configuration file name available: {}'.format(files))
    parser.add_argument('--rt_model', type=str, help='Path to checkpoint to restore training.')
    args = parser.parse_args()
    

    main()
