MOTION_SEGMENTATION_CONFIG = {
    'binary_thres': 150,
    'min_conected_comp_area': 49,
    'num_clusters_color_quantization': 5,
    'blur_kernel_size': 11,
    'k_brightnes_darkness_pixels': 100, #100
    'binary_thres_norm': 0.5,
    'k_best_components': 3,
    'plot_config':{
        'plot': False,
        'wait': 1000,
        'save_results': False, 
        'save_folder': None,
    },
}

TUBE_BUILD_CONFIG = {
    'train_mode': False, #Change
    'img_size': (224,224),
    'dataset_root': "/media/david/datos/Violence DATA/CCTVFights/frames",#"/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CCTVFights/frames", #Change
    'person_detections': '',
    'close_persons_rep': 10,#
    'temporal_window': 5, #5
    'min_iou_close_persons': 0.3,
    'jumpgap': 5,
    'min_window_len': 5,#3,
    'plot_config':{
        'debug_mode': False,
        'plot_tubes': False,
        'plot_wait_tubes': 100,
        'plot_wait_2':100,
        'save_results': False,
        'save_folder_debug': None,
        'save_folder_final': None,
    }
    
}
    