# RD-Detector

This repository contains the implementation of our work https://tvst.arvojournals.org/article.aspx?articleid=2802637, accepted to 
Translational Vision Science & Technology. 

## Installation
Please firstly install required libraries: pytorch, opencv, pandas, matplotlib, numpy, scikit-learn, scikit-image

```bash
python3 setup.py build_ext --inplace
```

## Usuage

### Training

This code inputs two folder path in which training and validation segmentation maps are located, extracts clinical explainable features  from the maps, carries out KNN traing whili finding the optimum k, and finilaly saves the knn-models as well as the cost function for finding the optimum k into given SAVE_DIR. 
```bash
python train.py TRAIN_DIR_PATH VAL_DIR_PATH SAVE_DIR
```


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