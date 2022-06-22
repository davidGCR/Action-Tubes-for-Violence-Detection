import add_libs
from utils.utils import get_torch_device, load_checkpoint, save_checkpoint
from utils.global_var import *
from utils.create_log_name import log_name_di_model

from configs.defaults import get_cfg_defaults
from datasets.make_dataset_handler import load_make_dataset 
from datasets.dataloaders import dataloaders_for_di_model
from models.resnet import ResNet
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import  lib.optimization_di as optimization_di
from lib.accuracy import calculate_accuracy_2

def main(h_path):
    # Setup cfg.
    cfg = get_cfg_defaults()
    cfg.merge_from_file(WORK_DIR / "configs/DI_MODEL.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    
    device = get_torch_device()
    make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=-1,
                                        train=True,
                                        category=2,
                                        shuffle=False)
    make_dataset_val = load_make_dataset(cfg.DATA,
                                    env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                    min_clip_len=-1,
                                    train=False,
                                    category=2,
                                    shuffle=False)
    
    # from debug.debug_didataset import debug_didataset
    # debug_didataset(cfg.DYNAMIC_IMAGE_DATASET, make_dataset_train, True)
    train_loader, val_loader, train_dataset, val_dataset = dataloaders_for_di_model(cfg, make_dataset_train, make_dataset_val)
    
    model = ResNet().to(device)
    params = model.parameters()
    exp_config_log = log_name_di_model(cfg)
    
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
        train_loss, train_acc, train_time = optimization_di.train(train_loader,
                                                                    epoch,
                                                                    cfg.SOLVER.EPOCHS, 
                                                                    model, 
                                                                    criterion, 
                                                                    optimizer, 
                                                                    device,
                                                                    calculate_accuracy_2)
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = optimization_di.val(val_loader,
                                                epoch, 
                                                model, 
                                                criterion,
                                                device,
                                                calculate_accuracy_2)
        scheduler.step(val_loss)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)
        
        if (epoch+1)%cfg.SOLVER.SAVE_EVERY == 0:
            save_checkpoint(model, 
                            cfg.SOLVER.EPOCHS, 
                            epoch, 
                            optimizer,
                            train_loss, 
                            os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))
    
if __name__=='__main__':
    h_path = HOME_WINDOWS
    torch.autograd.set_detect_anomaly(True)
    main(h_path)