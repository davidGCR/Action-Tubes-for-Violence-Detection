import torch
from torch.utils.data.dataloader import default_collate

def my_collate_2(batch):
    # batch = filter (lambda x:x is not None, batch)
    boxes = [item[0] for item in batch if item[0] is not None]
    images = [item[1] for item in batch if item[1] is not None]
    labels = [item[2] for item in batch if item[2] is not None]
    num_tubes = [item[3] for item in batch if item[3] is not None]
    batch = (boxes, images, labels, num_tubes)
    return default_collate(batch)

def my_collate(batch):
#     print('BATCH: ', type(batch), len(batch), len(batch[0]))
    boxes = [item[0] for item in batch if item[0] is not None]
    images = [item[1] for item in batch if item[1] is not None]
    labels = [item[2] for item in batch if item[2] is not None]
    # num_tubes = [item[3] for item in batch if item[3] is not None]
    paths = [item[3] for item in batch if item[3] is not None]
    # if len(batch[0]) == 6:
    #     key_frames = [item[5] for item in batch if item[5] is not None]

    key_frames = [item[4] for item in batch if item[4] is not None]

    # print('boxes:', type(boxes), len(boxes))
    # print('images:', type(boxes), len(images))
    # print('boxes[i]:', type(boxes[0]), boxes[0].size())
    # print('images[i]:', type(images[0]), images[0].size())
    
    boxes = torch.cat(boxes,dim=0)
    
    for i in range(boxes.size(0)):
        boxes[i][0] = i
        # print('--->', i, boxes[i])

    images = torch.cat(images,dim=0)
    labels = torch.tensor(labels)
    key_frames = torch.cat(key_frames,dim=0)
    # print('images cat:', images.size())
    # print('key_frames cat:', key_frames.size())
    # print('boxes cat:', boxes.size())
    return boxes, images, labels, paths, key_frames

    # if len(batch[0]) == 6:
    #     key_frames = torch.cat(key_frames,dim=0)

    #     return boxes, images, labels, paths, key_frames#torch.stack(labels, dim=0)
    # return boxes, images, labels, paths

def my_collate_video(batch):
#     print('BATCH: ', type(batch), len(batch), len(batch[0]))
    boxes = [item[0] for item in batch if item[0] is not None]
    images = [item[1] for item in batch if item[1] is not None]
    # labels = [item[2] for item in batch if item[2] is not None]

    key_frames = [item[2] for item in batch if item[2] is not None]

    # print('boxes:', type(boxes), len(boxes))
    # print('images:', type(boxes), len(images))
    # print('boxes[i]:', type(boxes[0]), boxes[0].size())
    # print('images[i]:', type(images[0]), images[0].size())
    
    boxes = torch.cat(boxes,dim=0)
    
    for i in range(boxes.size(0)):
        boxes[i][0] = i
        # print('--->', i, boxes[i])

    images = torch.stack(images,dim=0)
    key_frames = torch.stack(key_frames,dim=0)
    return boxes, images, key_frames