from utils.utils import get_number_from_string
from utils.tube_utils import bbox_iou_numpy
import numpy as np

def st_iou(tube, gt):
    """Compute spatio-temporal IOU

    Args:
        tube (dict): Tube
        gt (list): List of ground truth

    Returns:
        float: Spatio-temporal IOU
    """
    tgb = gt[0]['frame'] #gt begin
    tge = gt[-1]['frame'] #gt end

    tdb = get_number_from_string(tube['frames_name'][0]) #tube begin
    tde = get_number_from_string(tube['frames_name'][-1]) #tube end

    T_i = max(0,min(tge,tde)-max(tgb,tdb))
    # print('min(tge,tde): (tge,tde)=({},{}) = {}'.format(tge, tde, min(tge,tde)))
    # print('max(tgb,tdb): (tgb,tdb)=({},{}) = {}'.format(tgb, tdb, max(tgb,tdb)))
    # print('T_i: {}'.format(T_i))
    if T_i>0:
        T_u = max(tge, tde) - min(tgb, tdb) + 1
        # print('T_u: {}'.format(T_u))
        T_iou = T_i/T_u
        # print('T_iou: {}'.format(T_iou))
        frames_gt = [box_gt['frame'] for box_gt in gt]
        frames_dt = [int(get_number_from_string(f_name)) for f_name in tube['frames_name']]

        int_frames_numb = list(range(max(tgb,tdb), min(tge,tde)+1))
        # print('frames_gt: {}/{}'.format(frames_gt, len(frames_gt)))
        # print('frames_dt: {}/{}'.format(frames_dt, len(frames_dt)))
        # print('int_frames_numb: {}/{}'.format(int_frames_numb, len(int_frames_numb)))

        mask_gt = np.isin(np.array(frames_gt), np.array(int_frames_numb))
        int_find_gt = np.nonzero(mask_gt)[0]
        # print('int_find_gt: {}/{}'.format(int_find_gt, int_find_gt.shape))
        
        mask_dt = np.isin(np.array(frames_dt), np.array(int_frames_numb))
        int_find_dt = np.nonzero(mask_dt)[0]
        # print('int_find_dt: {}/{}'.format(int_find_dt, int_find_dt.shape))

        # assert int_find_gt.shape[0] == int_find_dt.shape[0], 'Error!!!'
        # ious = np.zeros(int_find_dt.shape[0])
        # for i in range(int_find_dt.shape[0]):
        #     gt_frame_name = frames_gt[int_find_gt[i]]
        #     dt_frame_name = frames_dt[int_find_dt[i]]
        #     assert gt_frame_name == dt_frame_name, 'Error: gt and dt inconsistency!!!-->{}!={}'.format(gt_frame_name, dt_frame_name)
        #     b1 = np.array([
        #         int(gt[int_find_gt[i]]['xmin']), 
        #         int(gt[int_find_gt[i]]['ymin']), 
        #         int(gt[int_find_gt[i]]['xmax']), 
        #         int(gt[int_find_gt[i]]['ymax'])]).reshape((1,4))#(1,4)
        #     b2 = np.array(tube['boxes'][int_find_dt[i]][:4]).reshape((1,4))
        #     ious[i] = bbox_iou_numpy(b1,b2)
        ious = np.zeros(len(frames_dt))
        for i in range(len(frames_dt)):
            # gt_frame_name = frames_gt[int_find_gt[i]]
            dt_frame_name = frames_dt[i]
            # assert gt_frame_name == dt_frame_name, 'Error: gt and dt inconsistency!!!-->{}!={}'.format(gt_frame_name, dt_frame_name)
            # print('---processing detected frame:', dt_frame_name)
            if dt_frame_name in frames_gt:
                idx = frames_gt.index(dt_frame_name)
                b1 = np.array([
                    int(gt[idx]['xmin']), 
                    int(gt[idx]['ymin']), 
                    int(gt[idx]['xmax']), 
                    int(gt[idx]['ymax'])]).reshape((1,4))#(1,4)
                b2 = np.array(tube['boxes'][int_find_dt[i]][:4]).reshape((1,4))
                ious[i] = bbox_iou_numpy(b1,b2)
            else:
                ious[i] = 0
        st_iou_ = T_iou*np.mean(ious)
        return st_iou_
    else:
        return 0