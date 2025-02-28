import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from Model import UNet
from torchvision import transforms
import torch.nn.functional as F
import joblib
from featureExtractor import featureExtract
from NoiseFilter import NoiseFiltering
import time
image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize() #Wait for all kernels in all streams on a CUDA device to complete.
    return time.time()



class RDDetector:
    def __init__(self, segmentatorPath, segmentatorType, classifierPath, input_size):
        
        if segmentatorType == 'TransUnet':
            patch_size = 16
            config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
            config_vit.n_classes = 4
            config_vit.n_skip = 3
            config_vit.patches.size = (patch_size, patch_size)
            config_vit.patches.grid = (int(input_size[0]/patch_size), int(input_size[1]/patch_size))
            self.model = ViT_seg(config_vit, img_size=input_size[0], num_classes=4).cuda()
        elif segmentatorType == 'UNet':
            self.model = UNet(1, 4, 64,
            True, False, 0.2)
        else:
            print('TODO')
        
        self.model.load_state_dict(torch.load(segmentatorPath))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.input_size = input_size
        self.knn, self.sc = self.loadModel(classifierPath)
        self.time_preprocess = 0
        self.time_inference = 0
        self.INFERENCETIME_TransUNET = 0
        self.INFERENCETIME_KNN = 0
        self.GPU2CPU_Time = 0
        self.NOISE_FILTERING = 0
        self.FEATURE_EXTRACTION = 0
        self.counter = 0

    def loadModel(self, model_path):
        # Load the saved model
        knn = joblib.load(model_path)

        print("Model loaded successfully!")
        
        scalar_path = model_path.replace('knn_model','scaler')
        # Load the saved scaler
        scaler = joblib.load(scalar_path)

        print("Scaler loaded successfully!")
        
        return [knn, scaler]
    
    def NoiseFiltering(self, img, thresh=150):
        return NoiseFiltering(img)
        
        
    def preprocess(self, img_org, input_size):
        if len(img_org.shape)==2:
            imgHeight, imgWidth = img_org.shape
            if imgHeight != input_size[0] or imgWidth != input_size[1]:
                img_input = zoom(img_org, (input_size[0] / imgHeight, input_size[1] / imgWidth), order=3)  
            else:
                img_input = img_org
        else:
            imgHeight, imgWidth, _ = img_org.shape
            if imgHeight != input_size[0] or imgWidth != input_size[1]:
                img_input = zoom(img_org, (input_size[0] / imgHeight, input_size[1] / imgWidth, 1), order=3)  
            else:
                img_input = img_org
            
        
        #z normalizization
        mean3d = np.mean(img_input, axis=(0,1))
        std3d = np.std(img_input, axis=(0,1))
        img_input = (img_input-mean3d)/std3d
        
        if len(img_org.shape)==2:
            img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            img_input = img_input.transpose((2, 0, 1))[::-1]
            print(img_input.shape)
            img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).cuda()
        
        return img_input
    
    
    def featureExtract(self, segMap):
        return featureExtract(segMap)
                        
    def infer(self, img_org):
        if self.counter>10:
            pred = self._inferTime(img_org)
        else:
            self.counter += 1
            pred = self._inferWOTime(img_org)
        return pred

    def _inferTime(self, img_org):
        s1 = time.time()
        img_input = self.preprocess(img_org, self.input_size)
        self.time_preprocess += time.time() - s1

        s2 = time.time()
        imgHeight, imgWidth = img_org.shape[0], img_org.shape[1]
        with torch.no_grad():
            s = time.time()
            outputs = self.model(img_input)
            probs = F.softmax(outputs, dim=1)
            out = torch.argmax(probs, dim=1).squeeze(0)
            self.INFERENCETIME_TransUNET+= time_synchronized()-s  # milliseconds
            s = time.time()
            out = out.to('cpu').numpy().astype(np.uint8)
            self.GPU2CPU_Time+= time_synchronized() - s

            if imgHeight != self.input_size[0] or imgWidth != self.input_size[1]:
                pred = zoom(out, (imgHeight / self.input_size[0], imgWidth / self.input_size[1]), order=0)
            else:
                pred = out
                
        s = time.time()
        pred = self.NoiseFiltering(pred, thresh=1500)
        self.NOISE_FILTERING+= time.time() - s

        s = time.time()
        X = self.featureExtract(pred)
        self.FEATURE_EXTRACTION+= time.time() - s

        s = time.time()
        # Use the loaded scaler to transform test data
        X_scalled = self.sc.transform(X)
    
        # Predicting the Test set results
        y_pred = self.knn.predict(X_scalled)
        self.INFERENCETIME_KNN += time.time() - s
        self.time_inference += time.time() - s2
        return y_pred
    
    def _inferWOTime(self, img_org):
        img_input = self.preprocess(img_org, self.input_size)

        imgHeight, imgWidth = img_org.shape[0], img_org.shape[1]
        with torch.no_grad():
            outputs = self.model(img_input)
            probs = F.softmax(outputs, dim=1)
            out = torch.argmax(probs, dim=1).squeeze(0)
            out = out.to('cpu', non_blocking=True).numpy().astype(np.uint8)

            if imgHeight != self.input_size[0] or imgWidth != self.input_size[1]:
                pred = zoom(out, (imgHeight / self.input_size[0], imgWidth / self.input_size[1]), order=0)
            else:
                pred = out
                
        pred = self.NoiseFiltering(pred, thresh=1500)
        X = self.featureExtract(pred)

        # Use the loaded scaler to transform test data
        X_scalled = self.sc.transform(X)
    
        # Predicting the Test set results
        y_pred = self.knn.predict(X_scalled)
        return y_pred

def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            if '_label' not in filename:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
    return natural_sort(image_names)

def main():
    
    test_path = '/home/ocaki13/projects/ultrasound/processed_data/ultrasoundSegmentationDatasetv2_resized_won/fold1/test/'
    image_list = get_image_list(test_path)
    segmentatorType = 'TransUnet' #Unet
    input_size = (800,800)
    segmentatorPath = '/home/ocaki13/projects/ultrasound/segmentationResults/exp4/TransUNet/us_exp4_wouaug_fold1/us_exp4_wouaug_fold1_seed629/epoch16.pt'
    classifierPath = '/home/ocaki13/projects/ultrasound/ClassificationOnSegments/fold1train/knn_model.pkl'

    rd = RDDetector(segmentatorPath, segmentatorType, classifierPath, input_size)

    times = 0
    counter = 0
    for imgPath in image_list:
        if counter>5:
            img = cv2.imread(imgPath, 0)
            s = time.time()
            a = rd.infer(img)
            times += time.time() - s
            print(imgPath, a)
        else:
            counter +=1
            img = cv2.imread(imgPath, 0)
            a = rd.infer(img)
            print(imgPath, a)
                
    if counter>5:
        average_time_ms = (times / (len(image_list)-counter)) * 1000
        print(f"Average time for whole pipeline: {average_time_ms:.3f} ms")

        average_time_ms = (rd.time_preprocess / (len(image_list)-rd.counter)) * 1000
        print(f"Average prerocess time: {average_time_ms:.3f} ms")
    
        average_time_ms = (rd.time_inference / (len(image_list)-rd.counter)) * 1000
        print(f"Average inference time: {average_time_ms:.3f} ms")
        
        average_time_ms = (rd.INFERENCETIME_TransUNET / (len(image_list)-rd.counter))*1000 
        print(f"Average TransUNet inference time: {average_time_ms:.3f} ms")
        
        average_time_ms = (rd.INFERENCETIME_KNN / (len(image_list)-rd.counter)) * 1000
        print(f"Average KNN inference time: {average_time_ms:.3f} ms")

        average_time_ms = (rd.FEATURE_EXTRACTION / (len(image_list)-rd.counter)) * 1000
        print(f"Feature extraction time: {average_time_ms:.3f} ms")
        
        average_time_ms = (rd.NOISE_FILTERING / (len(image_list)-rd.counter)) * 1000
        print(f"Noise filtering time: {average_time_ms:.3f} ms")
        
        average_time_ms = (rd.GPU2CPU_Time / (len(image_list)-rd.counter)) * 1000
        print(f"GPU CPU moving time: {average_time_ms:.3f} ms")

    
if __name__ == "__main__":
    main()