import joblib
import argparse
from utils import getTestList, print_results, evaluate_results, FeatureAndLabelExtraction
import pandas as pd
import os 

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('test_path', help='test folder path')
    ap.add_argument('model_path', help='model path')
    ap.add_argument('save_dir', help='folder path in which the results will be saved')
    args = ap.parse_args()
    return args

def loadModel(model_path):
    # Load the saved model
    knn = joblib.load(model_path)

    print("Model loaded successfully!")
    
    scalar_path = model_path.replace('knn_model','scaler')
    # Load the saved scaler
    scaler = joblib.load(scalar_path)

    print("Scaler loaded successfully!")
    
    return [knn, scaler]


def loadTestDict(testList):
    testDict = {}
    for sample in testList:
        currentSeed = sample.split('/')[-3].split('_')[-1]
        if currentSeed not in testDict:
            testDict[currentSeed] = []
        if 'PVD' in sample:
            testDict[currentSeed].append(sample)
    return testDict

def testKNN(testData, model_path, save_dir):

    
    print('Feature Extraction is under process..')
    dftest = FeatureAndLabelExtraction(testData)
    
    
    ####Feature Selection####
    selectedFeatures = ['projected1D_std',
                        'projected1D_max',
                        'areaRetinaFiltered']
    
    reduceddftest = pd.DataFrame(dftest, columns=selectedFeatures)

    #Create Matrix    
    X_test = reduceddftest.values
    y_test = dftest['label'].values
    
    print('Shape of data')
    print(X_test.shape)
    print(y_test.shape)
    
    knn, scaler = loadModel(model_path)

    # Use the loaded scaler to transform test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predicting the Test set results
    y_pred = knn.predict(X_test_scaled)
    
    evaluation_results=evaluate_results(y_pred, y_test)
    return evaluation_results
    
if __name__ == '__main__':
    args = parse_args()
    testData = getTestList(args.test_path)
    model_path = args.model_path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    bulk = False
    if bulk:
        testDict = loadTestDict(testData)
        resultsDict = {}
        for currentSeed in testDict:
            print('Test for {} is under process'.format(currentSeed))
            print(len(testDict[currentSeed]))
            evaluation_results=testKNN(testDict[currentSeed], model_path, save_dir)
            del evaluation_results['confusion_matrix']
            resultsDict[currentSeed] = evaluation_results
            
        # Convert the dictionary of dictionaries into a DataFrame
        results_df = pd.DataFrame.from_dict(resultsDict, orient='index')

        # Save the DataFrame to a CSV file
        results_df.to_csv(os.path.join(save_dir, "results.csv"), index_label="Seed")

        print("Results saved to results.csv")
        
            
    else:
        evaluation_results=testKNN(testData, model_path, save_dir)
        print_results(evaluation_results)

