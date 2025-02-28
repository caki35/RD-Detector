import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import getSampeList, FeatureAndLabelExtraction
import argparse
from QSL import similarityEuclidean, QSL_algorithm, Kolmogorov_Dmax
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--train', help='train folder path', required=True)
    ap.add_argument('-v', '--val', help='val folder path', required=True)
    ap.add_argument('-o', '--save_dir', help='folder path in which the results will be saved', required=True)
    args = ap.parse_args()
    return args



def KNNtrain(dftrain, dfval, selectedFeatures, n_optimum, save_dir):
    dftrain = pd.concat([dftrain, dfval])
    dftrain = dftrain.reset_index()
    
    ### Create data
    reduceddftrain = pd.DataFrame(dftrain, columns=selectedFeatures)
    X_train = reduceddftrain.values
    y_train = dftrain['label'].values
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    classifier = KNeighborsClassifier(n_neighbors = n_optimum, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    
    # Save the model to a file
    joblib.dump(classifier, os.path.join(save_dir, 'knn_model.pkl'))

    # Save the scaler
    joblib.dump(sc, os.path.join(save_dir,'scaler.pkl'))
    print("Model saved successfully!")
    
def main(args):
    trainData = getSampeList(args.train)
    valData = getSampeList(args.val)
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    print('Feature Extraction is under process..')
    dftrain = FeatureAndLabelExtraction(trainData)
    dfval = FeatureAndLabelExtraction(valData)
    
    ###Create RD Set###
    X_RD = pd.concat([dftrain[dftrain['label']==1], dfval[dfval['label']==1]])
    Y_RD = np.ones((len(X_RD),1))
    
    ###Create Normal Set###
    X_normal = pd.concat([dftrain[dftrain['label']==0], dfval[dfval['label']==0]])
    Y_normal = np.zeros((len(X_normal),1))  
    
    ####Feature Selection####
    selectedFeatures = ['projected1D_std',
                        'projected1D_max',
                        'areaRetinaFiltered']
    
    X_RD_reduced = pd.DataFrame(X_RD, columns=selectedFeatures)
    X_normal_reduced= pd.DataFrame(X_normal, columns=selectedFeatures)

    #Extract Data Matrix
    Xrd = X_RD_reduced.values
    Xnormal = X_normal_reduced.values  
    
    print('RD C1 X: {}'.format(Xrd.shape))
    print('RD C1 Y: {}'.format(Y_RD.shape))
    print('Control C0 X: {}'.format(Xnormal.shape))
    print('Control C0 Y: {}'.format(Y_normal.shape))
    
    #merge them into one dataset for QSL
    X=np.concatenate((Xrd, Xnormal))
    Y=np.concatenate((Y_RD, Y_normal))
    
    # Feature Scaling
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)

    #create similarity matrix
    sm_df = similarityEuclidean(Xscaled)

    #put labels into a dataframe 
    labels_df= pd.DataFrame(data=Y,columns=["label"])
    
    #Run QSL
    results =QSL_algorithm(sm_df,labels_df,[1,40])

    #optimum n
    n_optimum = results['n_optimum']
    
    #cost function of QSL
    cost=results['cost']
    n_list=results['n_list']
    f0=results['f0']
    f1=results['f1']

    #plot and save cost function
    plt.plot(n_list,cost,'-bo')
    index = np.where(n_list == results['n_optimum'])
    plt.plot(results['n_optimum'], cost[index[0], index[1]], '-ro')
    plt.grid()
    plt.xlabel('n')
    plt.ylabel('E(n)')
    plt.title('Cost Function')
    plt.savefig(os.path.join(save_dir,'QSLcost_function.png'), dpi=300)
    
    KNNtrain(dftrain, dfval, selectedFeatures, n_optimum, save_dir)
if __name__ == '__main__':
    args = parse_args()

    main(args)