import os
import re
import pandas as pd 
import numpy as np
import cv2
from skimage import measure
from tqdm import tqdm
import shutil
from skimage.measure import label
import time
import torch
#control  PVD  RD
label_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]

def create_rgb_mask(mask, label_colors):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = label_colors[0]
    rgb_mask[mask == 2] = label_colors[1]
    rgb_mask[mask == 3] = label_colors[2]
    # rgb_mask[mask == 4] = label_colors[3]

    return rgb_mask

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize() #Wait for all kernels in all streams on a CUDA device to complete.
    return time.time()

def getSampeList(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in ['.png'] and '_label' in filename:
                image_names.append(apath)
    return natural_sort(image_names)

def getTestList(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in ['.png'] and '_pred' in filename:
                image_names.append(apath)
    return natural_sort(image_names)

def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def ANOVAtest(df):
    from scipy.stats import f_oneway
    results = {}
    for c in df.columns:
        if c == 'label':
            continue
        if c == 'paths':
            continue
        normaldist = df[df['label']==0][c]
        rddist = df[df['label']==1][c]
        currentf = f_oneway(normaldist, rddist).statistic
        currentp = f_oneway(normaldist, rddist).pvalue
        results[c] = {'f_score':currentf, 'p_value':currentp}
    sortedRes = sorted(results.items(), key=lambda x: x[1]['f_score'], reverse=True)
    indexs = []
    fscores = []
    ps = []
    for ele in sortedRes:
        indexs.append(ele[0])
        fscores.append(ele[1]['f_score'])
        ps.append(ele[1]['p_value'])

    featureANOVAdf = pd.DataFrame(list(zip(fscores, ps)),index = indexs,
               columns =['f_score', 'ps'])
    return featureANOVAdf

label_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]

def create_rgb_mask(mask):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = label_colors[0]
    rgb_mask[mask == 2] = label_colors[1]
    rgb_mask[mask == 3] = label_colors[2]
    # rgb_mask[mask == 4] = label_colors[3]
    return cv2.cvtColor(rgb_mask,cv2.COLOR_BGR2RGB)

def evaluate_results(Y_predicted,Y_actual):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0,len(Y_actual)):
        if Y_predicted[i] == Y_actual[i]:
            if Y_actual[i] == 1:
                tp += 1
            else:
                tn +=1
        else:
            if Y_actual[i] == 1:
                fn += 1
            else:
                fp +=1
    

    # Construct the confusion matrix as a 2x2 numpy array
    cm = np.array([[tn, fp],  # Row 1: Actual Negatives
                [fn, tp]]) # Row 2: Actual Positives

    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(
        data=cm,
        index=["Actual Negatives", "Actual Positives"],
        columns=["Predicted Negatives", "Predicted Positives"]
    )

    accuracy=(tp+tn)/(tp+fp+tn+fn)
    epsilon = 1e-9
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)
    
    evaluation_results = {"confusion_matrix": cm_df,
                                  "accuracy": round(accuracy*100,2),
                                  "precision": round(precision*100,2),
                                "recall":round(recall*100,2),
                                  "f1Score":round(f1*100,2)}
    
    return evaluation_results


def print_results(evaluation_results):
    print(evaluation_results['confusion_matrix'])
    print("")
    print("Accuracy=",round(evaluation_results['accuracy'],2))
    print("Recall=", round(evaluation_results['recall'],2))
    print("Precision=",round(evaluation_results['precision'],2))
    print("F1 Score=",round(evaluation_results['f1Score'],2))
    print(round(evaluation_results['precision'],2),
          round(evaluation_results['recall'],2),
          round(evaluation_results['f1Score'],2))
    print("{}\t{}\t{}".format(
        round(evaluation_results['precision'],2),
          round(evaluation_results['recall'],2),
          round(evaluation_results['f1Score'],2)
    ))
    
    
def featureExtractProjection1D(segimg):
    RetinaArea = (segimg==2).astype(np.uint8)
    projectedList = []
    for c in range(0,RetinaArea.shape[1]):
        if (np.sum(RetinaArea[:,c]) != 0):
            indices = np.where(RetinaArea[:,c] ==1)
            diff = np.diff(indices)
            projectedList.append(np.count_nonzero(diff != 1)+1)
    
    return projectedList
        
def extractFalseSamples(y_pred, y, dftest):
    if not os.path.exists('FN'):
        os.mkdir('FN')
    else:
        shutil.rmtree('FN')
        os.mkdir('FN')
    if not os.path.exists('FP'):
        os.mkdir('FP')
    else:
        shutil.rmtree('FP')
        os.mkdir('FP')   
    for i in range(0, len(y_pred)):
        ### False Positive
        if y_pred[i] ==1 and y[i]==0:
            imgPath = dftest.at[i, 'paths']
            #read sample
            sampleName = imgPath.split('/')[-1].split('_pred.png')[0]
            img = cv2.imread(imgPath.split('_pred.png')[0], 0)
            #write sample
            #cv2.imwrite(os.path.join('FP',sampleName), img)
            
            #read prediction
            imgpred = cv2.imread(imgPath, 0)
            rgb_mask_pred = create_rgb_mask(imgpred)
            rgb_mask_pred = cv2.cvtColor(rgb_mask_pred, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(os.path.join('FP',imgPath.split('/')[-1]), imgpred_rgb)
            
            #read gt
            samplePath = imgPath.split('_pred.png')[0]
            gtPath = samplePath[:samplePath.rfind('.')] + '_label.png'
            imgGT = cv2.imread(gtPath, 0)
            rgb_mask_GT = create_rgb_mask(imgGT)
            rgb_mask_GT = cv2.cvtColor(rgb_mask_GT, cv2.COLOR_BGR2RGB)
            #write gt
            #cv2.imwrite(os.path.join('FP',imgPath.split('/')[-1]), imgGT)
            
            seperater = np.zeros([img.shape[0], 15, 3], dtype=np.uint8)
            seperater.fill(155)
            save_img_bin = np.hstack(
                [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), seperater, rgb_mask_GT, seperater, rgb_mask_pred])
            cv2.imwrite(os.path.join('FP', sampleName), save_img_bin)
                        
        ### False Negative
        elif y_pred[i] ==0 and y[i]==1:
            imgPath = dftest.at[i, 'paths']
            #read sample
            sampleName = imgPath.split('/')[-1].split('_pred.png')[0]
            img = cv2.imread(imgPath.split('_pred.png')[0], 0)
            #write sample
            #cv2.imwrite(os.path.join('FN',sampleName), img)
            
            #read prediction
            imgpred = cv2.imread(imgPath, 0)
            rgb_mask_pred = create_rgb_mask(imgpred)
            rgb_mask_pred = cv2.cvtColor(rgb_mask_pred, cv2.COLOR_BGR2RGB)
            #write prediction
            #cv2.imwrite(os.path.join('FN',imgPath.split('/')[-1]), imgpred_rgb)
            
            #read gt
            samplePath = imgPath.split('_pred.png')[0]
            gtPath = samplePath[:samplePath.rfind('.')] + '_label.png'
            imgGT = cv2.imread(gtPath, 0)
            rgb_mask_GT = create_rgb_mask(imgGT)
            rgb_mask_GT = cv2.cvtColor(rgb_mask_GT, cv2.COLOR_BGR2RGB)
            #write gt
            #cv2.imwrite(os.path.join('FN',gtPath.split('/')[-1]), imgGT)
            seperater = np.zeros([img.shape[0], 15, 3], dtype=np.uint8)
            seperater.fill(155)
            save_img_bin = np.hstack(
                [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), seperater, rgb_mask_GT, seperater, rgb_mask_pred])
            cv2.imwrite(os.path.join('FN', sampleName), save_img_bin)
    
proplist = ['area',
 'area_bbox',
 'area_convex',
 'area_filled',
 'axis_major_length',
 'axis_minor_length',
 #'bbox',
 'centroid',
 'centroid_local',
 #'centroid_weighted',
 #'centroid_weighted_local',
 #'coords',
 'eccentricity',
 'equivalent_diameter_area',
 'euler_number',
 'extent',
 'feret_diameter_max',
 #'image',
 #'image_convex',
 #'image_filled',
 #'image_intensity',
 'inertia_tensor',
 'inertia_tensor_eigvals',
 #'intensity_max',
 #'intensity_mean',
 #'intensity_min',
 #'label',
 'moments',
 'moments_central',
 'moments_hu',
 #'moments_normalized',
 #'moments_weighted',
 #'moments_weighted_central',
 #'moments_weighted_hu',
 #'moments_weighted_normalized',
 'orientation',
 'perimeter',
 'perimeter_crofton',
 #'slice',
 'solidity']


def NoiseFiltering(img, thresh=150):
    binary_img = np.zeros_like(img)
    binary_img[img == 2] = 1
    label_img = label(binary_img)
    label_list = list(np.unique(label_img))
    for lbl in label_list:
        if (label_img == lbl).sum() < thresh:
            img[label_img == lbl] = 0
    return img
    
def moveToLeft(binary_img,tx):
    y_indices = np.where(binary_img==1)[0]
    x_indices = np.where(binary_img==1)[1]
    new_indices =[y_indices, x_indices-tx]
    translated_img = np.zeros_like(binary_img)

    for y, x in [new_indices]:
        translated_img[y,x] = 1
    return translated_img

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC.astype(np.uint8)
    
def filterRetina(pred):
    #take sclera as binary image
    binary_img = np.zeros_like(pred)
    binary_img[pred == 3] = 1
    binary_img = getLargestCC(binary_img)
    #y center
    y_center = int(np.mean(np.unique(np.where(binary_img==1)[0])))

    #calculate center row (how much go left)
    height_center_row = np.where(binary_img[y_center,:]==1)
    x_center = int(np.mean(height_center_row))
    tx = len(height_center_row[0])
    #translate sclare towards left
    translated_image = moveToLeft(binary_img,tx)
    #remove overlapped pixels
    pred[translated_image==1]=0
    return pred


def featureExtract(segMap):
    RetinaArea = (segMap==2).astype(np.uint8)
    projectedList = []
    for c in range(0,RetinaArea.shape[1]):
        if (np.sum(RetinaArea[:,c]) != 0):
            indices = np.where(RetinaArea[:,c] ==1)
            diff = np.diff(indices)
            projectedList.append(np.count_nonzero(diff != 1)+1)

    projected1D_max = max(projectedList)
    projected1D_std = np.std(projectedList)
    segimg_filtered = filterRetina(segMap.copy())
    binary_img_retina = np.zeros_like(segimg_filtered)
    binary_img_retina[segimg_filtered == 2] = 1
    areaRetina = np.sum(binary_img_retina)

    return np.array([projected1D_std, projected1D_max, areaRetina], dtype=np.float64).reshape(1, -1)

    
def FeatureAndLabelExtraction(segMaps):
    class_list = []
    features = []
    paths = []
    projected1D_max = []
    projected1D_std = []
    areaRetinaFiltered = []
    for currentsegMap in tqdm(segMaps):
        paths.append(currentsegMap)
        segimg = cv2.imread(currentsegMap,0)
        #noise filtering
        segimg = NoiseFiltering(segimg, thresh=1500)
                
        projected1D = featureExtractProjection1D(segimg)
        projected1D_max.append(max(projected1D))
        projected1D_std.append(np.std(projected1D))
        foldername = currentsegMap.split('/')[-2]
        if 'RD' in foldername:
            class_list.append(1)
        else:
            class_list.append(0)   
        props = measure.regionprops_table(segimg, properties=proplist)
        numberLabel = len(np.unique(segimg))
        currentFeature = []
        #optic nerve in ocular image
        # retina(2) #sclera(3)
        if numberLabel == 4:
            for key in props:
                currentProps = props[key]
                #pass optic nerve features
                for i in range(1,3):
                    currentFeature.append(currentProps[i])
        else:
            for key in props:
                currentProps = props[key]
                for i in range(0,2):
                    currentFeature.append(currentProps[i])
        features.append(currentFeature)
        
        #new filtered retina feature
        segimg_filtered = filterRetina(segimg.copy())
        binary_img_retina = np.zeros_like(segimg_filtered)
        binary_img_retina[segimg_filtered == 2] = 1
        areaRetina = np.sum(binary_img_retina)
        areaRetinaFiltered.append(areaRetina)


    new_heads= []
    for currentattribute in list(props.keys()):
        new_heads.append(currentattribute + '_Retina')
        new_heads.append(currentattribute + '_Sclera')
    
    #construct dataframe for dataset
    X = np.array(features)
    df = pd.DataFrame(data=X, columns= new_heads)
    df['projected1D_max'] = projected1D_max
    df['projected1D_std'] = projected1D_std
    df['areaRetinaFiltered'] = areaRetinaFiltered
    #add classes
    df['label'] = class_list
    #add paths
    df['paths'] = paths

    return df
