
from os import path

from torch._C import _LegacyVariableBase
from transformations.data_aug.data_aug import *
# from model_transformations import i3d_video_transf, resnet_transf

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.make_dataset import *
from datasets.make_UCFCrime import *
from datasets.tube_dataset import *
from datasets.CCTVFights_dataset import * 
from datasets.make_cctvfights import make_CCTVFights_dataset, make_CCTVFights_dataset_clips

from utils.vizualize_batch import *
from utils.dataset_utils import read_JSON_ann
import matplotlib.pyplot as plt

def test_dataset(cfg):
    ann_file  = (cfg.UCFCRIME_DATASET.TRAIN_ANNOT_ABNORMAL, cfg.UCFCRIME_DATASET.TRAIN_ANNOT_NORMAL)# if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
    home_path = cfg.DATA.ROOT
    make_dataset = MakeUCFCrime(
            root=os.path.join(home_path, cfg.UCFCRIME_DATASET.ROOT), 
            sp_abnormal_annotations_file=os.path.join(home_path, cfg.DATA.SPLITS_FOLDER,'UCFCrime', ann_file[0]), 
            sp_normal_annotations_file=os.path.join(home_path, cfg.DATA.SPLITS_FOLDER, 'UCFCrime', ann_file[1]), 
            action_tubes_path=os.path.join(home_path, cfg.DATA.ACTION_TUBES_FOLDER, cfg.UCFCRIME_DATASET.NAME),
            train=True,
            ground_truth_tubes=True)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)  
    inputs_config = {
        'input_1': {
            'type': 'rgb',
            # 'spatial_transform': i3d_video_transf()['train'],
            'spatial_transform': Compose(
                [
                    ClipRandomHorizontalFlip(), 
                    # ClipRandomScale(scale=0.2, diff=True), 
                    ClipRandomRotate(angle=5),
                    # ClipRandomTranslate(translate=0.1, diff=True),
                    NumpyToTensor()
                ],
                probs=[1, 1]
                ),
            'temporal_transform': None
        },
        # 'input_2': {
        #     'type': 'rgb',
        #     'spatial_transform': resnet_transf()['val'],
        #     'temporal_transform': None
        # }
        'input_2': {
            'type': 'dynamic-image',
            'spatial_transform': transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                norm
            ]),
            'temporal_transform': None
        }
    }
    # train_dataset = TubeDataset(frames_per_tube=16, 
    #                         make_function=make_dataset,
    #                         max_num_tubes=1,
    #                         train=True,
    #                         dataset='UCFCrime_Reduced',
    #                         random=True,
    #                         tube_box=UNION_BOX,
    #                         config=TWO_STREAM_INPUT_train,
    #                         key_frame=DYNAMIC_IMAGE_KEYFRAME)

    # cfg.TUBE_DATASET.MAKE_FN = make_dataset
    cfg.TUBE_DATASET.DATASET = cfg.UCFCRIME_DATASET.NAME
    # cfg.TUBE_DATASET.DATALOADERS_DICT = TWO_STREAM_INPUT_train
    cfg.TUBE_DATASET.FRAMES_STRATEGY = MIDDLE_FRAMES
    cfg.TUBE_DATASET.BOX_STRATEGY = MIDDLE_BOX
    cfg.TUBE_DATASET.KEYFRAME_STRATEGY = DYNAMIC_IMAGE_KEYFRAME
    train_dataset = TubeDataset(
                                cfg.TUBE_DATASET,
                                make_dataset,
                                inputs_config,
                                UCFCrimeReduced_DATASET
                                )
    
    # for i in range(len(train_dataset)):
    #     data = train_dataset[i]
    #     bboxes, video_images, label, num_tubes, path, key_frames = data
    #     if os.path.split(path)[1]=='Assault027_x264':
    #         print(i)
    #         break
    # random.seed(34)
    for i in range(1):
        bboxes, video_images, label, num_tubes, path, key_frames = train_dataset[40]
        print('\tpath: ', path)
        print('\tvideo_images: ', type(video_images), video_images.size())
        print('\tbboxes: ', bboxes.size())
        print('\tkey_frames: ', type(key_frames), key_frames.size())

        frames_numpy = video_images.permute(0,2,3,4,1)
        frames_numpy = frames_numpy.cpu().numpy().astype('uint8')
        bboxes_numpy = bboxes.cpu().numpy()[:,1:5] #remove id and to shape (n,4)\
        key_frames_numpy = key_frames.permute(0,2,3,1)
        key_frames_numpy = key_frames_numpy.cpu().numpy()
        
        print('\nframes_numpy: ', frames_numpy.shape)
        print('bboxes_numpy: ', bboxes_numpy)
        print('key_frames_numpy: ', key_frames_numpy.shape)

        for j in range(frames_numpy.shape[0]): #iterate over batch
            
            bboxes_numpy = [bboxes_numpy] * 16
            plot_clip(frames_numpy[j], bboxes_numpy, (4,4))
            plot_keyframe(key_frames_numpy[j], bboxes_numpy[0])


def test_tube_dataset(train_dataset, val_dataset):
    final_tube_boxes, video_images, labels, path, key_frames = train_dataset[0]
    print('video_images: ', video_images.size())
    print('key_frames: ', key_frames.size())
    print('final_tube_boxes: ', final_tube_boxes,  final_tube_boxes.size())
    print('labels: ', labels)

    # print('final_tube_boxes: ', final_tube_boxes, final_tube_boxes.size())
    # print('key_frames: ', len(key_frames_raw), type(key_frames_raw[0]))
    # final_tube_boxes = final_tube_boxes.cpu().numpy()
    # for k, im in enumerate(key_frames_raw):
    #     # rect = final_tube_boxes[k,1:5]
    #     # print('\n',k,'\n')
    #     # print(k, rect)
    #     # print("input crop: ",rect[0],rect[1],rect[2],rect[3], rect.reshape(1,-1))
    #     # plot_keyframe(np.array(im), rect.reshape(1,-1))
    #     # im_c = im.crop((rect[0],rect[1],rect[2],rect[3]))
    #     # im_c = im_c.resize(im.size)
    #     # plt.imshow(np.array(im_c))
    #     # plt.show()

    #     plt.imshow(np.array(im))
    #     plt.show()


def test_cctvfights_datasets(cfg, transforms_config_train, transforms_config_val):
    root = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CCTVFights/frames/fights"
    json_file = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CCTVFights/groundtruth_modified.json"
    pers_annotations_folder = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/CCTVFights/fights"
    data = read_JSON_ann(json_file)
    print(data["version"])

    make_fn = make_CCTVFights_dataset_clips(root, json_file, pers_annotations_folder, "validation")

    paths, labels, indices, tmp_annotations, pers_annotations = make_fn()
    print('paths: ', len(paths))
    print('labels: ', len(labels))
    print('indices: ', len(indices))
    print('tmp_annotations: ', len(tmp_annotations))
    print('pers_annotations: ', len(pers_annotations))

    print('======Statictics=====')
    count_fight = sum(map(lambda x : x["label"]=="Fight", tmp_annotations))
    count_nonfight = sum(map(lambda x : x["label"]=="NonFight", tmp_annotations))
    print('fight instances: ', count_fight)
    print('nonfight instances: ', count_nonfight)

    # idx = 100
    # print('paths[]: ', paths[idx])
    # print('labels[]: ', labels[idx])
    # print('indices[]: ', indices[idx])
    # print('tmp_annotations[]: ', tmp_annotations[idx])
    
    dataset_ = ClipDataset(
                            cfg,
                            32, 
                            1, 
                            paths, 
                            labels, 
                            tmp_annotations, 
                            pers_annotations, 
                            False, 
                            transforms_config_train, #transforms.ToTensor(), 
                            True
                            )
    for i in range(len(dataset_)):
        path, label, tmp_annotation, pers_annotation, clip, sampled_clip, video_images, video_boxes, keyframes = dataset_[i]
        print("\npath: {}/label: {}".format(path, label))
        print("tmp_annot: {}".format(tmp_annotation))
        print("pers_annotation: {}".format(pers_annotation))
        print("clip: {}/{}, sampled_clip: {}/{}".format(clip, len(clip), sampled_clip, len(sampled_clip)))
        print("video_images: ", video_images.size())
        print("video_boxes: ", video_boxes.size())
        print("keyframes: ", keyframes.size())
        if i == 1:
            break


    