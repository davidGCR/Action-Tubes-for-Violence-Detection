import torch
from torch.utils.tensorboard.summary import video
from utils.utils import AverageMeter, get_number_from_string
import numpy as np
import time
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

#validation log  videos
from datasets.CCTVFights_dataset import SequentialDataset
from torch.utils.data import DataLoader
from datasets.collate_fn import my_collate
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

def train(_loader, _epoch, _num_epochs, _model, _criterion, _optimizer, _device, _num_tubes, _accuracy_fn, _verbose=False):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    batch_time = AverageMeter()
    end_time = time.time()
    # For roc_auc
    y_true, y_pred = torch.zeros(0,dtype=torch.long, device='cpu'), torch.zeros(0,dtype=torch.long, device='cpu')

    loop = tqdm(enumerate(_loader), total=len(_loader), leave=False)
    for batch_index, data in loop:
        boxes, video_images, labels, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None

        # print('video_images: ', video_images.size())
        # print('key_frames: ', key_frames.size())
        # print('boxes: ', boxes,  boxes.size())
        # print('labels: ', labels, labels.size())

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(video_images, key_frames, boxes, _num_tubes)
        # print('labels: ', labels, labels.size())
        # print('outs: ', outs, outs.size())
        # _, preds = torch.max(outs, dim=1)
        probabilities = F.softmax(outs, dim=1)[:, 1]
        
        # y_true.append(labels.cpu().numpy())
        # y_pred.append(probabilities.cpu().detach().numpy())
        y_true = torch.cat([y_true, labels.view(-1).cpu()])
        y_pred = torch.cat([y_pred, probabilities.view(-1).cpu()])
        
        # print("preds: ", preds, preds.size())
        # print("probabilities: ", probabilities, probabilities.size())
        
        #loss
        loss = _criterion(outs, labels)
        #accuracy
        acc = _accuracy_fn(outs, labels)
        
        
        # meter
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if _verbose:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    _epoch,
                    i + 1,
                    len(_loader),
                    loss=losses,
                    acc=accuracies,
                    batch_time=batch_time
                )
            )
        
        loop.set_description(f"Epoch [{_epoch}/{_num_epochs}]")
        loop.set_postfix(loss=loss.item(), acc=acc, time=batch_time.val)

    # print("y_true: ", y_true.size())
    # print("y_pred: ", y_pred.size())

    roc_auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().detach().numpy())
    # print("roc_auc: ", roc_auc)

    train_loss = losses.avg
    train_acc = accuracies.avg
    time_ = batch_time.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss:.4f}\t'
        'Acc(train): {acc:.3f}\t'
        'ROC_AUC(train): {roc_auc:.4f}\t'
        'Time: {tim:.3f}'.format(_epoch, loss=train_loss, acc=train_acc, roc_auc=roc_auc, tim=time_*batch_time.counter)
    )
    return train_loss, train_acc, time_

def val(_loader, _epoch, _model, _criterion, _device, _num_tubes, _accuracy_fn):
    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    y_true, y_pred = torch.zeros(0,dtype=torch.long, device='cpu'), torch.zeros(0,dtype=torch.long, device='cpu')
    # Init loggers
    # starter = torch.cuda.Event(enable_timing=True)
    # ender = torch.cuda.Event(enable_timing=True)
    # timings=np.zeros((len(_loader),1))
    
    for k, data in tqdm(enumerate(_loader), total=len(_loader), leave=False):
        boxes, video_images, labels, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None
        # no need to track grad in eval mode
        with torch.no_grad():
            # starter.record()
            outputs = _model(video_images, key_frames, boxes, _num_tubes)
            # ender.record()
            
            # wait for GPU sync
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # timings[k] = curr_time
        
            loss = _criterion(outputs, labels) if _criterion is not None else 0
            acc = _accuracy_fn(outputs, labels)
            
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            y_true = torch.cat([y_true, labels.view(-1).cpu()])
            y_pred = torch.cat([y_pred, probabilities.view(-1).cpu()])
        if _criterion is not None:
            losses.update(loss.item(), outputs.shape[0])
        accuracies.update(acc, outputs.shape[0])
    roc_auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().detach().numpy())
    val_loss = losses.avg
    val_acc = accuracies.avg


    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss:.4f}\t'
        'Acc(val): {acc:.3f}\t'
        'ROC_AUC(val): {roc_auc:.4f}\t'.format(_epoch, loss=val_loss, acc=val_acc, roc_auc=roc_auc)
    )
    return val_loss, val_acc

def fps(_loader, _epoch, _model, _device, _num_tubes):
    # print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # Init loggers
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings=np.zeros((len(_loader),1))
    
    print("Warming...")
    repetitions = 1000
    input = torch.randn(4,3,16,224,224, dtype=torch.float).to(_device)
    key_frames = torch.randn(4,3,224,224, dtype=torch.float).to(_device)
    boxes = torch.randn(4,5, dtype=torch.float).to(_device)
    for rep in range(repetitions):
        _ = _model(input, key_frames, boxes, _num_tubes)
        torch.cuda.synchronize()
    
    for k, data in tqdm(enumerate(_loader), total=len(_loader), leave=False):
        boxes, video_images, labels, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None
        # no need to track grad in eval mode
        with torch.no_grad():
            start_time = time.time()
            starter.record()
            outputs = _model(video_images, key_frames, boxes, _num_tubes)
            ender.record()
            
            # wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[k] = curr_time
            video_time = time.time()-start_time
            frame_time = video_time/(4*16)
            video_fps = 1.0/frame_time
            print('\n \tCurr Time: {:.4f} miliseconds\tVideo time: {:.4f} seconds \tFrame time: {:.4f} seconds \tVideo fps: {:.4f}'.format(
                curr_time, 
                video_time, 
                frame_time,
                video_fps))
    
    # calculate mean and standard deviation
    mean_syn = np.sum(timings) / (len(_loader)*16) #time in ms for each frame
    std_syn = np.std(timings)
    fps = 1/mean_syn

    print(
        'Epoch: [{}]\t'
        'Time(val): {time:.4f} ms\t'
        'FPS(val): {fps:.4f} FPS'.format(_epoch, time=mean_syn, fps=fps)
    )


def val_map(_loader, _epoch, _model, _criterion, _device, _num_tubes):
    # print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    # accuracies = AverageMeter()
    # Initialize the prediction and label lists(tensors)
    ypred = torch.zeros(0,dtype=torch.long, device='cpu')
    ytrue = torch.zeros(0,dtype=torch.long, device='cpu')
    
    for _, data in enumerate(_loader):
        boxes, video_images, labels, ntubes, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        
        # print('video_images: ', video_images.size())
        # print('key_frames: ', key_frames.size())
        # print('boxes: ', boxes,  boxes.size())
        # print('labels: ', labels, labels.size())
        # print('ntubes: ', ntubes, len(ntubes))
        
        ytrue = torch.cat([ytrue, labels.view(-1).cpu()])
        if ntubes[0] == 0:
            ypred = torch.cat([ypred, torch.tensor([0])])
            continue
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(video_images, key_frames, boxes, _num_tubes)
            outputs = torch.unsqueeze(outputs, dim=0)
            _, preds = torch.max(outputs, 1)
            # print('outputs: ', outputs,  outputs.size())
            # print('labels: ', labels, labels.size())
            # print('preds: ', preds, preds.size())
            ypred = torch.cat([ypred, preds.view(-1).cpu()])
    return ytrue, ypred
    
    print('ytrue: ', ytrue, ytrue.size())
    print('ypred: ', ypred, ypred.size())
    print('lens: ', ytrue.size(), ypred.size())
    
    map = average_precision_score(ytrue.cpu().numpy(), ypred.cpu().numpy())
    print("MAP: ", map)
    exit()

    val_loss = losses.avg
    val_acc = 0
    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss:.4f}\t'
        'Acc(val): {acc:.3f}'.format(_epoch, loss=val_loss, acc=val_acc)
    )
    return val_loss, val_acc


def train_2d_branch(
    _loader,
    _epoch, 
    _model,
    _criterion, 
    _optimizer, 
    _device, 
    _config,
    _accuracy_fn, 
    _verbose=False):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes = boxes.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(key_frames, boxes, _config.num_tubes)
        #loss
        # print('labels: ', labels, labels.size(),  outs, outs.size())
        loss = _criterion(outs, labels)
        #accuracy
        acc = _accuracy_fn(outs, labels)
        
        # meter
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()
        if _verbose:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    _epoch,
                    i + 1,
                    len(_loader),
                    loss=losses,
                    acc=accuracies
                )
            )
        
    train_loss = losses.avg
    train_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss:.4f}\t'
        'Acc(train): {acc:.3f}'.format(_epoch, loss=train_loss, acc=train_acc)
    )
    return train_loss, train_acc

def val_2d_branch(_loader, _epoch, _model, _criterion, _device, _config, _accuracy_fn):

    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    for _, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes = boxes.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(key_frames, boxes, _config.num_tubes)
            loss = _criterion(outputs, labels)
            acc = _accuracy_fn(outputs, labels)

        losses.update(loss.item(), outputs.shape[0])
        accuracies.update(acc, outputs.shape[0])
    val_loss = losses.avg
    val_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss:.4f}\t'
        'Acc(val): {acc:.3f}'.format(_epoch, loss=val_loss, acc=val_acc)
    )
    

    return val_loss, val_acc

def validate_long_videos(cfg, make_fn, clip_len, tubes_path, transforms,_epoch , **kwargs):
        print('validation at epoch: {}'.format(_epoch))
        paths, frame_rates, tmp_annotations, person_det_files = make_fn()
        ypred = torch.zeros(0,dtype=torch.long, device='cpu')
        ytrue = torch.zeros(0,dtype=torch.long, device='cpu')
        for j, (path, frame_rate, tmp_annot, pers_detect_annot) in tqdm(enumerate(zip(paths, frame_rates, tmp_annotations, person_det_files)), total=len(paths), leave=False):
            # print(j, path)
            # print("tmp_annot: ", tmp_annot)
            # print("pers_detect_annot: ", pers_detect_annot)
            # print("frame_rate: ", frame_rate)
            dataset = SequentialDataset(cfg=cfg,
                                        seq_len=clip_len, 
                                        tubes_path=tubes_path, 
                                        pers_detect_annot=pers_detect_annot, 
                                        annotations=tmp_annot, 
                                        video_path=path, 
                                        frame_rate=frame_rate, 
                                        transforms=transforms)

            loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=my_collate
                            )
            
            ytrue_video, ypred_video = val_map(loader,
                                                kwargs["epoch"], 
                                                kwargs["model"], 
                                                kwargs["criterion"],
                                                kwargs["device"],
                                                kwargs["num_tubes"])
            ytrue = torch.cat([ytrue, ytrue_video.view(-1).cpu()])
            ypred = torch.cat([ypred, ypred_video.view(-1).cpu()])
        # print('ytrue: ', ytrue.size())
        # print('ypred: ', ypred.size())
        # print('lens: ', ytrue.size(), ypred.size())
        
        map = average_precision_score(ytrue.cpu().numpy(), ypred.cpu().numpy())
        print(
            'Epoch: [{}]\t'
            'map(val): {map:.4f}\t'.format(_epoch, map=map)
        )
        return map