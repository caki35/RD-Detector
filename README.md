# RD-Detector

This repository contains the implementation of our study, ["Automated Detection of Retinal Detachment Using Deep Learning-Based Segmentation on Ocular Ultrasonography Images"](https://tvst.arvojournals.org/article.aspx?articleid=2802637)
 published in Translational Vision Science & Technology.

## Installation
Before running the code, ensure that the required libraries are installed:: pytorch, opencv, pandas, matplotlib, numpy, scikit-learn, scikit-image

```bash
python3 setup.py build_ext --inplace
```

## Usage

### Train


The training script takes two folder paths containing segmentation maps for training and validation. It extracts clinical explainable features from the segmenpation maps, performs K-Nearest Neighbors (KNN) training while determining the optimum k, and saves both the trained KNN model and the cost function for k selection in given SAVE_DIR.

```bash
python train.py -t TRAIN_DIR_PATH -v VAL_DIR_PATH -o SAVE_DIR
```

### Testing & Inference

The testing and inference scripts process test images in a given test folder using the paths to a trained segmentation model and a classifier model from the previous step.
- The test script evaluates classification performance by comparing predictions against ground-truth labels (derived from folder names).
- The inference script generates segmentation outputs without classification results and prints the classification prediction as well.

```bash
python3 test.py -i TEST_DIR_PATH -s SEGMENTATION_MODEL_PATH -c ML_CLASSIFER_MODEL_PATH -o SAVE_DIR
```

```bash
python3 inference.py -i TEST_DIR_PATH -s SEGMENTATION_MODEL_PATH -c ML_CLASSIFER_MODEL_PATH -o SAVE_DIR
```

You can train your own segmentation model using the following repository:
https://github.com/caki35/UNet-Torch

## Citation

Please consider citing our paper if you find it useful. 
```
@article{10.1167/tvst.14.2.26,
    author = {Caki, Onur and Guleser, Umit Yasar and Ozkan, Dilek and Harmanli, Mehmet and Cansiz, Selahattin and Kesim, Cem and Akcan, Rustu Emre and Merdzo, Ivan and Hasanreisoglu, Murat and Gunduz-Demir, Cigdem},
    title = {Automated Detection of Retinal Detachment Using Deep Learning-Based Segmentation on Ocular Ultrasonography Images},
    journal = {Translational Vision Science & Technology},
    volume = {14},
    number = {2},
    pages = {26-26},
    year = {2025},
    month = {02},
    issn = {2164-2591},
    doi = {10.1167/tvst.14.2.26},
    url = {https://doi.org/10.1167/tvst.14.2.26},
    eprint = {https://arvojournals.org/arvo/content\_public/journal/tvst/938704/i2164-2591-14-2-26\_1740577359.0831.pdf},
}
```