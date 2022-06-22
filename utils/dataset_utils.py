from torchvision import transforms
import numpy as np
from PIL import Image
import json

def tensor2PIL(t):
    im = transforms.ToPILImage()(t).convert("RGB")
    return im

def PIL2tensor(img):
    im = transforms.ToTensor()(img)
    return im

def PIL2numpy(img):
    return np.array(img)

def imread(path, resize=None):
    try:
        with Image.open(path) as img:
            image = img.convert('RGB')
            if resize is not None:
                image = image.resize(resize)
            return image
    except Exception as e:
        print("\nOops!", e.__class__, "occurred. Path: ", path)

def read_JSON_ann(json_file):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        return data

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
        decodedArray = sorted(decodedArray, key = lambda i: i['id'])
        return decodedArray
    
def filter_data_without_tubelet(paths, labels, annotations):
        indices_2_remove = []
        for index in range(len(paths)):
            path = paths[index]
            label = labels[index]
            annotation = annotations[index]
            tubelets = JSON_2_tube(annotation)
            if len(tubelets) == 0:
                # print('No tubelets at: ',path)
                indices_2_remove.append(index)

        paths = [paths[i] for i in range(len(paths)) if i not in indices_2_remove]
        labels = [labels[i] for i in range(len(labels)) if i not in indices_2_remove]
        annotations = [annotations[i] for i in range(len(annotations)) if i not in indices_2_remove]
        return paths, labels, annotations


# def JSON_2_tube(json_file):
#     """
#     """
#     with open(json_file, "r") as read_file:
#         decodedArray = json.load(read_file)
#         # print("decoded Array:", type(decodedArray), len(decodedArray))
        
#         for f in decodedArray:
#             for i, box in  enumerate(f['boxes']):
#                 f['boxes'][i] = np.asarray(f['boxes'][i])
#         # print(decodedArray[0])
#         decodedArray = sorted(decodedArray, key = lambda i: i['id'])
#         return decodedArray

def check_no_tubes(make_function):
    paths, labels, annotations = make_function()
    videos_no_tubes = []
    for i, ann in enumerate(annotations):
        tubes = JSON_2_tube(ann)
        if len(tubes)==0:
            videos_no_tubes.append(paths[i])
    
    return videos_no_tubes