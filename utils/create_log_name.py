def log_name(cfg):
    str_ = "{}_{}_CV({})_TF({})_usingGT({})_numTubes({})_framesXtube({})_framesStrat({})_boxStrat({})_keyframeInput({})_loss({})_opt({})_lr({})_epochs({})".format(
        cfg.MODEL.NAME+'-'+cfg.MODEL._HEAD.NAME+'-'+cfg.MODEL._3D_BRANCH.NAME+'+'+cfg.MODEL._2D_BRANCH.NAME+'-3dRoi-'+str(cfg.MODEL._3D_BRANCH.WITH_ROIPOOL)+'-2dRoi-'+str(cfg.MODEL._2D_BRANCH.WITH_ROIPOOL),
        cfg.DATA.DATASET,
        cfg.DATA.CV_SPLIT,
        cfg.MODEL.TRANSF_LEARNING.ACTIVE,
        cfg.DATA.LOAD_GROUND_TRUTH,
        cfg.TUBE_DATASET.NUM_TUBES,
        cfg.TUBE_DATASET.NUM_FRAMES,
        cfg.TUBE_DATASET.FRAMES_STRATEGY,
        cfg.TUBE_DATASET.BOX_STRATEGY,
        cfg.TUBE_DATASET.KEYFRAME_STRATEGY,
        cfg.SOLVER.CRITERION,
        cfg.SOLVER.OPTIMIZER.NAME,
        cfg.SOLVER.LR,
        cfg.SOLVER.EPOCHS
    )
    return str_

def log_name_di_model(cfg):
    str_ = "{}_{}_CV({})_numClips({})_clipLen({})_clipStride({})_loss({})_opt({})_lr({})_epochs({})".format(
        cfg.MODEL.NAME,
        cfg.DATA.DATASET,
        cfg.DATA.CV_SPLIT,
        cfg.DYNAMIC_IMAGE_DATASET.NUM_CLIPS,
        cfg.DYNAMIC_IMAGE_DATASET.CLIP_LEN,
        cfg.DYNAMIC_IMAGE_DATASET.CLIP_STRIDE,
        cfg.SOLVER.CRITERION,
        cfg.SOLVER.OPTIMIZER.NAME,
        cfg.SOLVER.LR,
        cfg.SOLVER.EPOCHS
    )
    return str_