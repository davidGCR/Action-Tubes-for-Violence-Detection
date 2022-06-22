import torch
from torch.utils.tensorboard.summary import video
from utils.utils import AverageMeter, get_number_from_string
import numpy as np
import time
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from datasets.CCTVFights_dataset import SequentialDataset
from torch.utils.data import DataLoader
from datasets.collate_fn import my_collate
from sklearn.metrics import average_precision_score

def train(_loader, _epoch, _num_epochs, _model, _criterion, _optimizer, _device, _accuracy_fn):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    batch_time = AverageMeter()
    end_time = time.time()

    loop = tqdm(enumerate(_loader), total=len(_loader), leave=False)
    for batch_index, data in loop:
        images, labels = data
        images = images.to(_device)
        labels = labels.to(_device)

        # print('images: ', images.size())
        # print('labels: ', labels.size())

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(images)
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

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        loop.set_description(f"Epoch [{_epoch}/{_num_epochs}]")
        loop.set_postfix(loss=loss.item(), acc=acc, time=batch_time.val)

    train_loss = losses.avg
    train_acc = accuracies.avg
    time_ = batch_time.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss:.4f}\t'
        'Acc(train): {acc:.3f}\t'
        'Time: {tim:.3f} mins'.format(_epoch, loss=train_loss, acc=train_acc, tim=time_*batch_time.counter/60)
    )
    return train_loss, train_acc, time_

def val(_loader, _epoch, _model, _criterion, _device, _accuracy_fn):
    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    for _, data in tqdm(enumerate(_loader), total=len(_loader), leave=False):
        images, labels = data
        images = images.to(_device)
        labels = labels.to(_device)
        # print('images: ', images.size())
        # print('labels: ', labels.size())
        
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(images)
            loss = _criterion(outputs, labels) if _criterion is not None else 0
            acc = _accuracy_fn(outputs, labels)
        if _criterion is not None:
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