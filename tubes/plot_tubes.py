import os
import numpy as np
from utils.visual_utils import draw_boxes, imread, color
import cv2
from pathlib import Path

def plot_tubes(paths, tubes, wait=200, save_folder=None):
    """Plot action tubes

    Args:
        paths (list): List of image paths
        tubes (list): List of action tubes
        wait (int, optional): Plot wait. Defaults to 200.
    """
    # print('save_folder in plot: ', save_folder)
    images_to_video = []
    colors = []
    video_name = Path(paths[0]).parents[0].name
    if len(tubes)>1:
        for l in range(len(tubes)):
            b_color = (
                    np.random.randint(0,255), 
                    np.random.randint(0,255), 
                    np.random.randint(0,255)
                    )
            colors.append(b_color)
    else:
        colors.append((0,255,0)) #default green
    for index in range(len(paths)):
        print(index)
        frame = np.array(imread(paths[index]))
        frame_name = Path(paths[index]).name

        #Persons
        # pred_boxes = self.video_detections[t]['pred_boxes'] #real bbox
        # if pred_boxes.shape[0] != 0:
        #     image = draw_boxes(image,
        #                         pred_boxes[:, :4],
        #                         # scores=pred_boxes[:, 4],
        #                         # tags=pred_tags_name,
        #                         line_thick=1, 
        #                         line_color='white')
        box_tubes = []
        tube_ids = []
        tube_scores = []
        
        for l in range(len(tubes)):
            foundAt = True if frame_name in tubes[l]['frames_name'] else False
            if foundAt:
                idx = tubes[l]['frames_name'].index(frame_name)
                bbox = tubes[l]['boxes'][idx]
                box_tubes.append(bbox)
                tube_ids.append(tubes[l]['id'])
                tube_scores.append(tubes[l]['score'])
            
        if len(box_tubes)>0:
            box_tubes = np.array(box_tubes)
            # print('iamge shape: ', image.shape)
            frame = draw_boxes(frame,
                                box_tubes[:, :4],
                                # scores=tube_scores,
                                ids=tube_scores,
                                line_thick=2, 
                                line_color=colors)
        images_to_video.append({
            "image": frame,
            "name": frame_name
            })

        # if save_folder is not None:
        #     print('savinggggg')
        #     filename = Path(video_name)/frame_name
        #     print("frame to save: ",filename)
            
        # cv2.namedWindow('FRAME'+frame_name,cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('FRAME'+frame_name, (600,600))
        frame = cv2.resize(frame, (600,600))
        # cv2.imshow('FRAME'+frame_name, frame)
        cv2.imshow('FRAME', frame)
        # key = cv2.waitKey(wait)
        key = cv2.waitKey(10)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
        
        
    
    if save_folder is not None:
        folder = Path(save_folder)/Path(video_name)
        folder.mkdir(parents=True, exist_ok=True)
            
        for i, img_data in enumerate(images_to_video):
            
            filename = folder/img_data['name']
            # print("frame to save: ",filename)
            cv2.imwrite(str(filename), img_data['image'])