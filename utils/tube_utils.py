import numpy as np
import json
from json import JSONEncoder
import cv2
import os

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def videoDetections_2_JSON(output_path: str, video_detections: list):
    """
    Args:
        output_path: Folder to save JSON's
        video_detections: list pf Dictionaries. Each dict has the format:
            frame_detection = {
                            "fname": os.path.split(img_path)[1], #frame name
                            "video": one_video, #video name
                            "split": sp, #train/val
                            "pred_boxes":  pred_boxes[:, :5], #numpy array
                            "tags": pred_tags_name #list of labels ['person', 'person', ...]
                        }
    """
    with open(output_path, 'w') as fout:
        json.dump(video_detections , fout, cls=NumpyArrayEncoder)

def JSON_2_videoDetections(json_file):
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            f['pred_boxes'] = np.asarray(f['pred_boxes'])
        # print(decodedArray[0])
        return decodedArray

def tube_2_JSON(output_path: str, tube: list):
    """
    {
        'frames_name': [video_detections[t]['fname']], #list of
        'boxes':[merge_pred_boxes[b,:]], #list of numpy arrays
        'len': 1, #length of tube
        'id': b, #id tube
        'foundAt': [t], #list of list, each list are the frames in each tube
        'lastfound': 0 #diff between current frame and last frame in path
    }
    """
    with open(output_path, 'w') as fout:
        json.dump(tube , fout, cls=NumpyArrayEncoder)

def JSON_2_tube(json_file):
    """
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            for i, box in  enumerate(f['boxes']):
                f['boxes'][i] = np.asarray(f['boxes'][i])
        # print(decodedArray[0])
        return decodedArray


def JSON_2_videoDetections(json_file):
    """
    Load Spatial detections from  a JSON file.
    Return a List with length frames. 
    An element in the list contain a dict with the format:
    {
        'fname': 'frame1.jpg',
        'video': '0_DzLlklZa0_3',
        'split': 'train/Fight',
        'pred_boxes': array[],
        'tags': list(['person', ...])
    }
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            f['pred_boxes'] = np.asarray(f['pred_boxes'])
        # print(decodedArray[0])
        return decodedArray


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    # print('box1: ', box1.shape, '--box2: ', box2.shape)
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def merge_bboxes(bbox1, bbox2, flac):
    """
    return:
        array of shape (1,5)
    """

    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    # s = max(bbox1[4], bbox2[4])
    # s = bbox1[4] + bbox2[4]
    s = flac
    # print('joined score: ', bbox1[4], bbox2[4], s)
    return np.array([x1, y1, x2, y2, s]).reshape(1,-1)

def merge_bboxes_numpy(bboxes, flac):
    """
    input:  
        (N,5) array
    return:
        array of shape (1,5)
    """
    # print('bboxes',bboxes, bboxes.shape)
    # print('jmin: ', np.amin(bboxes, axis=0))
    # print('jmax: ', np.amax(bboxes, axis=0))

    x1 = np.amin(bboxes, axis=0)[0]
    y1 = np.amin(bboxes, axis=0)[1]
    x2 = np.amax(bboxes, axis=0)[2]
    y2 = np.amax(bboxes, axis=0)[3]
    # s = max(bbox1[4], bbox2[4])
    s = flac
    # print('joined score: ', bbox1[4], bbox2[4], s)
    return np.array([x1, y1, x2, y2, s]).reshape(1,-1)

def create_video(images, image_folder, video_name, save_frames=False):
    height, width, layers = images[0].shape
    video_name = os.path.join(image_folder, video_name)
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    # Appending the images to the video one by one
    for i, image in enumerate(images): 
        if save_frames:
            cv2.imwrite(image_folder + '/'+str(i+1)+'.jpg', image)
        video.write(image) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

def save_video_tubes(folder_out, video_path, tubes):
    """[summary]

    Args:
        folder_out (str): Path to Tubes folder.
        video_path (str): Path of the video folder.
        tubes (list): List of Tubes (dict)
    """
    parts = video_path.split('/')
    out_file = os.path.join(folder_out, parts[-2], parts[-1]+'.json')

    if not os.path.isdir(os.path.join(folder_out, parts[-2])): #Create folder of split
        os.makedirs(os.path.join(folder_out, parts[-2]))
    
    if os.path.exists(out_file):
        print('Already done!!!')
        return
    
    tube_2_JSON(output_path=out_file, tube=tubes)