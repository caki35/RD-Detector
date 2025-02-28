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
from skimage.measure import label 
#from utils import NoiseFiltering
from utils import featureExtract, create_rgb_mask, time_synchronized, label_colors
import argparse
import time
image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']

class RDDetector:
    def __init__(self, segmentatorPath, segmentatorType, classifierPath, input_size, save_dir):
        
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
        if torch.cuda.is_available():
            self.model.cuda()
        self.input_size = input_size
        self.knn, self.sc = self.loadModel(classifierPath)
        self.save_dir = save_dir

    def loadModel(self, model_path):
        # Load the saved model
        knn = joblib.load(model_path)

        print("Model loaded successfully!")
        
        scalar_path = model_path.replace('knn_model','scaler')
        # Load the saved scaler
        scaler = joblib.load(scalar_path)

        print("Scaler loaded successfully!")
        
        return [knn, scaler]
    
    def NoiseFiltering(self, img, thresh=1500):
        return NoiseFiltering(img, thresh)
          

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
            img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).cuda()
        
        return img_input
    
    def featureExtract(self, segMap):
        return featureExtract(segMap)
                        
    def infer(self, img_org, img_name):
        img_input = self.preprocess(img_org, self.input_size)

        imgHeight, imgWidth = img_org.shape[0], img_org.shape[1]
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_input)
            probs = F.softmax(outputs, dim=1)
            out = torch.argmax(probs, dim=1).squeeze(0)
            out = out.cpu().detach().numpy().astype(np.uint8)

            if imgHeight != self.input_size[0] or imgWidth != self.input_size[1]:
                pred = zoom(out, (imgHeight / self.input_size[0], imgWidth / self.input_size[1]), order=0)
            else:
                pred = out
            

        rgb_mask_pred = create_rgb_mask(pred)
        cv2.imwrite(os.path.join(self.save_dir, img_name), rgb_mask_pred)
        pred = self.NoiseFiltering(pred, thresh=1500)

        X = self.featureExtract(pred)

        # Use the loaded scaler to transform test data
        X_scalled = self.sc.transform(X)
    
        # Predicting the Test set results
        y_pred = self.knn.predict(X_scalled)

        return y_pred[0]
    


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

def getGoldLabel(path):
    folderName = path.split('/')[-2]
    if 'RD' in folderName:
        return 1
    else:
        return 0

def main():
    
    
    parser = argparse.ArgumentParser(
    description="RD-Detector Test - Inference on images in a given folder, compare with the label in the folder names and save results")
    parser.add_argument('-i', '--input_folder_tast_path', required=True)
    parser.add_argument('-s', '--segmentation_model_path', required=True)
    parser.add_argument('-c', '--ml_classifier_path', required=True)
    parser.add_argument('-o', '--output_folder_path', required=True)
    args = parser.parse_args()
    
    segmentatorType = 'TransUnet' #Unet
    input_size = (800,800)

    test_path = args.input_folder_tast_path
    image_list = get_image_list(test_path)
    segmentatorPath = args.segmentation_model_path
    classifierPath = args.ml_classifier_path
    save_dir = args.output_folder_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    rd = RDDetector(segmentatorPath, segmentatorType, classifierPath, input_size, save_dir)

    total_inference_time = 0
    counter = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for imgPath in image_list:
        imgName = imgPath.split('/')[-1]
        lbl = getGoldLabel(imgPath)
        img = cv2.imread(imgPath, 0)
        if counter>5:             
            s = time.time()
            pred = rd.infer(img, imgName)
            total_inference_time += time_synchronized()-s  # second
        else:
            pred = rd.infer(img, imgName)
            counter += 1

        print(imgPath, pred)
        if lbl == pred:
            if lbl == 0:
                tn +=1
            else:
                tp +=1
        else:
            if lbl == 1:
                fn +=1
            else:
                fp +=1

                    
    precision = round(tp/(fp+tp),4)*100
    recall = round(tp/(fn+tp),4)*100
    f1 = round(2*precision*recall/(precision+recall),4)
    accuracy = round((tp+tn)/(tp+tn+fn+fp),4)*100
    print('Precision: ',precision)
    print('Recal: ', recall)
    print('f1: ', f1)
    print('accuracy', accuracy)
    if counter>5:             
        total_inference_time /= len(image_list)-counter
        print('total inference time {:.4f} sec'.format(total_inference_time))
    
    f = open("{}/results.txt".format(save_dir), 'w')
    f.write('The number of Total Image: ' +
            ' ' + str(len(image_list)) + '\n')
    f.write('\n')
    f.write('False Negatives: ' + ' ' + str(fn) + '\n')
    f.write('False Positives: ' + ' ' + str(fp) + '\n')
    f.write('True Positives:: ' + ' ' + str(tp) + '\n')
    f.write('True Negatives:: ' + ' ' + str(tn) + '\n')
    f.write('\n')

    f.write('Precision: ' + ' ' + str(precision) + '\n')
    f.write('Recall: ' + ' ' + str(recall) + '\n')
    f.write('F1 Score: ' + ' ' + str(f1) + '\n')
    f.write('Accuracy: ' + ' ' + str(accuracy) + '\n')

    f.write('total inference time: ' + ' ' + str(round(total_inference_time,4))  + ' sec \n')
    f.write('\n')
                


    
if __name__ == "__main__":
    main()