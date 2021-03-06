import numpy as np
import torch
from utils.dataset_utils import imread

from PIL import Image
import cv2

class DynamicImage():
    def __init__(self, output_type="pil", savePath=None, vizualize=False):
        self.savePath = savePath
        self.output_type = output_type
        self.vizualize = vizualize
    
    def __to_tensor__(self, img):
        img = img.astype(np.float32)/255.0
        t = torch.from_numpy(img)
        t = t.permute(2,0,1)
        return t
    
    def __read_imgs__(self, pths):
        # for p in pths:
        #     _, v = os.path.split(p)
        #     print(v)
        frames = [imread(p) for p in pths]
        # frames = [np.array(imread(p)) for p in pths]
        return frames

    def __call__(self, frames):
        if isinstance(frames[0], str):
            frames = self.__read_imgs__(frames)
        elif torch.is_tensor(frames):
            frames = frames.numpy()
            frames = [f for f in frames]
        seqLen = len(frames)
        if seqLen < 2:
            print('No se puede crear DI con solo un frames ...', seqLen)
        
        # print(len(frames))
        # for f in frames:
        #     print('--', f.shape)
        try:
            frames = np.stack(frames, axis=0)
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print(len(frames))
            for f in frames:
                print('--', f.size)

        fw = np.zeros(seqLen)  
        for i in range(seqLen): #frame by frame
            fw[i] = np.sum(np.divide((2 * np.arange(i + 1, seqLen + 1) - seqLen - 1), np.arange(i + 1, seqLen + 1)))
        # print('Di coeff=',fw)
        fwr = fw.reshape(seqLen, 1, 1, 1)  #coeficiebts
        sm = frames*fwr
        sm = sm.sum(0)
        sm = sm - np.min(sm)

        sm = sm.astype(np.float64)

        sm = 255 * sm / (np.max(sm)+0.00001)
        img = sm.astype(np.uint8)
        ##to PIL image
        imgPIL = Image.fromarray(np.uint8(img))
        if self.vizualize:
            # imgPIL.show()
            frame = cv2.resize(img, (600,600))
            # cv2.imshow('FRAME'+frame_name, frame)
            cv2.imshow('Dynamic Iamge', frame)
            key = cv2.waitKey(0)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 
        if self.savePath is not None:
            imgPIL.save(self.savePath)
            
        if self.output_type == "ndarray":
            return img
        elif self.output_type == "pil":
            return imgPIL
        elif self.output_type == "tensor":
            return self.__to_tensor__(img)