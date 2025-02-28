import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import itertools
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
import os 
def similarityEuclidean(Xscaled):
    distance_matrix=euclidean_distances(Xscaled, Xscaled)

    #turn distance into similarity
    similarity_matrix=np.zeros((distance_matrix.shape))
    for i in range(0,similarity_matrix.shape[0]):
        for j in range(0,similarity_matrix.shape[1]):
            similarity_matrix[i,j]=1/(1+distance_matrix[i,j])

    sm_df = pd.DataFrame(data=similarity_matrix)
    
    return sm_df

def distance_calculate(similarity_matrix, j, label_data):

    xi = similarity_matrix.iloc[:, [j]].sort_values(
        by=similarity_matrix.columns[j], ascending=False)

    labels = []
    for index in xi.index.values:
        labels.append(label_data['label'][index])
    #append it as a new column into xi

    return labels


def posterior_calculate(di, n):
    '''
    Inputs:
    di --> the binary numpy array consist of labels of all instance 
           0. index = label of querry instance
           Other labels are sorted by similarity between the instances to which they belong and query instance
    n  --> how many isntances are used from C0 and C1 seperately to construct reference sets
    ----------------------------------------------------------------------------------------------
    Output:
    f1 --> posterior probabilty of query instance of belonging to C1 set.
    '''
    #identify k*
    k_prime = min([l[0] for l in enumerate(di) if l[1] == 1][-n],
                  [l[0] for l in enumerate(di) if l[1] == 0][-n])

    Pr_y_Ek = di[k_prime]  # P(y|k_prime-1)=1(y)
    for k in range(k_prime-1, 0, -1):
        if(di[k] == 1):  # the total number of samples with label 1 beyond this point
            l_n = di[k:].count(1)
        if(di[k] == 0):  # the total number of samples with label 0 beyond this point
            l_n = di[k:].count(0)

        Pr_y_Ek = (n/l_n)*di[k] + (1-(n/l_n))*Pr_y_Ek

    return Pr_y_Ek


def QSL_algorithm(similarity_matrix, label_data, nrange):
    '''
    it's takes three inputs:
    similarity_matrix --> Similarity values between instances
                          It must be in dataframe format. 
                          Its indices and columns name must be same
    label_data        --> It's also dataframe where the last column consist of label of instances
                          The index name, or numbers in label_data must be same with the similarity_matrix 
    nrane             --> the range of number of sample are taken each class to construct reference sets
    --------------------------------------------------------------------------
    Returns dictionary output that contains following elements:
    n_list: evaluated list of n (1,2,...,nmax)
    cost: calculated cost of which we use each n in n_list
    f0: posterior probabilities of which each samples belong to C0 when we use the n with minimum cost
    f1: posterior probabilities of which each samples belong to C1 when we use the n with minimum cost
    '''

    #number of instance
    m = len(label_data)

    #the number of instance in C1
    number_of_C1 = sum(label_data['label'])

    #excract labels from dict
    instances_labels = label_data['label'].values

    #initialize posterior probabilites
    f0 = np.zeros((1, m))
    f1 = np.zeros((1, m))

    n_min = nrange[0]
    n_max = nrange[1]

    #list of the n that we will use to calculate posterior probabilities
    n_list = np.arange(n_min, n_max+1, 1)
    #initialize cost array that in which each cost for each n will keep
    cost = np.zeros((1, len(n_list)))
    for n in n_list:
        print("Calculating Posterior Probabilities for n={}".format(n))
        for i in tqdm(range(0, m)):
            # similarity between i. insantance and all other instances
            di = distance_calculate(similarity_matrix, i, label_data)
            # posterior probability of which i. instance belong C1
            f1[0][i] = posterior_calculate(di, n)
        f0 = 1-f1
        print('')
        # calculate overlap cost over the C1
        cost_C1 = 4*np.sum(np.multiply(np.multiply(f0, f1), instances_labels))
        # calculate overlap cost over the C0
        cost_C0 = 4*np.sum(np.multiply(np.multiply(f0, f1),
                           abs(instances_labels-1)))

        #normalize cost over C0 to avoid class-inbalance
        #penalize greater n to achieve generalization
        cost[0][n-1] = cost_C1+((number_of_C1/(m-number_of_C1)*cost_C0))+(2*n)

    n_list = np.reshape(n_list, (1, n_max))  # (6,) -> (1,6)

    #Find optimum n
    n_optimum = n_list[0, np.argmin(cost)]

    #by using optimum, calculate optimum posterior probabilities
    f0_optimum = np.zeros((1, m))
    f1_optimum = np.zeros((1, m))
    print("Calculating Posterior Probabilities for optimum n={}".format(n_optimum))
    for i in tqdm(range(0, m)):
        # similarity between i. insantance and all other instances
        di = distance_calculate(similarity_matrix, i, label_data)
        # posterior probabilit of which i. instance belong C1
        f1_optimum[0][i] = posterior_calculate(di, n_optimum)
    f0_optimum = 1-f1_optimum

    results = {"n_list": n_list,
               "cost": cost,
               "n_optimum": n_optimum,
               "f0": f0_optimum,
               "f1": f1_optimum}
    return results


def get_minimum(cost_dict):
    exp_dict = cost_dict.copy()
    min_n1 = min(exp_dict.keys(), key=(lambda k: exp_dict[k]))
    exp_dict.pop(min_n1, None)
    min_n2 = min(exp_dict.keys(), key=(lambda k: exp_dict[k]))
    return min_n1, min_n2


def n_list_generator(min_n1, min_n2, n_steps):
    n_list = [min_n1, min_n2]
    n_list.sort()

    while(n_list[-2]+n_steps < max(n_list)):
        n_list.append(n_list[-2]+n_steps)
        n_list.sort()

    return n_list



def Kolmogorov_Dmax(posteriordist_df):

    C1 = posteriordist_df[posteriordist_df["label"] == 1]
    C0 = posteriordist_df[posteriordist_df["label"] == 0]

    #copy the samples and their posterior distributions into new dataframe
    Kolmogorov_df = posteriordist_df.copy()

    #add new columns that wil be used to calculate CDF and Dmax into this dataframe
    zero_column = np.zeros((len(Kolmogorov_df), 1))
    Kolmogorov_df["F1_C0"] = zero_column  # the column of CDF for C0
    Kolmogorov_df["F1_C1"] = zero_column  # the column of CDF for C1
    # the column of difference between two CDF
    Kolmogorov_df["Dmax"] = zero_column

    #sort the posterior distributions the samples in C1 and C0 together
    sorted_df = Kolmogorov_df.sort_values(by=['f1'])

    #give equal probability of occuring to each samples in two dataset
    p0 = 1/len(C0)
    p1 = 1/len(C1)

    #inport to numpy matrix for calculations
    KSD_table = sorted_df.values

    Dmax = 0
    T = 0

    #first sample
    if KSD_table[0][0] == 0:
        KSD_table[0][2] = KSD_table[0][2]+p0
    else:
        KSD_table[0][3] = KSD_table[0][3]+p1
    KSD_table[0][4] = abs(KSD_table[0][3]-KSD_table[0][2])

    #CDF calculation
    for i in range(1, len(KSD_table)):
        if KSD_table[i][0] == 0:  # curent sample in C0
            KSD_table[i][2] = KSD_table[i-1][2]+p0
            KSD_table[i][3] = KSD_table[i-1][3]
        else:  # curent sample in C1
            KSD_table[i][3] = KSD_table[i-1][3]+p1
            KSD_table[i][2] = KSD_table[i-1][2]
        KSD_table[i][4] = abs(KSD_table[i][3]-KSD_table[i][2])

        if KSD_table[i][4] >= Dmax:  # Comparing Dmax
            Dmax = KSD_table[i][4]  # If bigger change
            F1_C0_T = KSD_table[i][2]  # Save CDF of C0
            F1_C1_T = KSD_table[i][3]  # Save CDF of C1
            T = KSD_table[i][1]  # Save current posterior as threshold

    CDF_C0 = KSD_table[:, 2]
    CDF_C1 = KSD_table[:, 3]
    posterior_samples = KSD_table[:, 1]

    ks_results = {"Dmax": Dmax,
                  "T": T,
                  "CDF_C0": CDF_C0,
                  "CDF_C1": CDF_C1,
                  "posterior_samples": posterior_samples,
                  "F1_C0_T": F1_C0_T,
                  "F1_C1_T": F1_C1_T}

    return ks_results


def plot_results(posteriordist_df, ks_results, save_dir):
    C1 = posteriordist_df[posteriordist_df["label"] == 1]
    C0 = posteriordist_df[posteriordist_df["label"] == 0]
    Dmax = ks_results['Dmax']
    T = ks_results['T']
    F1_C0 = ks_results['CDF_C0']
    F1_C1 = ks_results['CDF_C1']
    f1_all = ks_results['posterior_samples']

    F1_C0_T = ks_results['F1_C0_T']
    F1_C1_T = ks_results['F1_C1_T']

    #### Saving Results ####
    import seaborn as sns
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(8.1)
    fig.set_figwidth(10.8)

    count, bin_edges = np.histogram(posteriordist_df.iloc[:, 1])

    #create Dmax
    y_dmax = np.arange(F1_C1_T, F1_C0_T, 0.01)
    x_dmax = np.full(y_dmax.shape, T)

    sns.distplot(C1.iloc[:, 1], axlabel=False, 
                 rug=True, bins=bin_edges, ax=axs[0, 1])
    axs[0, 1].title.set_text('Distribution of f1 in C1')
    axs[0, 1].set_xlim(0, 1)
    axs[0, 1].grid(True)

    sns.distplot(C0.iloc[:, 1], axlabel=False,
                 rug=True, bins=bin_edges, ax=axs[1, 1])
    axs[1, 1].title.set_text('Distribution of f1 in C0')
    axs[1, 1].set_xlim(0, 1)
    axs[1, 1].grid(True)
    axs[1, 1].axvline(x=T, ymin=0, ymax=1, label='T', c='m')
    axs[1, 1].annotate("T= "+str(round(T, 3)), xy=(T, 5))
    axs[1, 1].set_xlabel('P(C1|x)')

    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, 0])

    axbig.plot(f1_all, F1_C0, 'r', label='C0')
    axbig.plot(f1_all, F1_C1, 'b', label='C1')
    axbig.axvline(x=T, ymin=0, ymax=1, label='T = '+str(round(T, 3)), c='m')
    axbig.plot(x_dmax, y_dmax, 'y', label='Dmax = '+str(round(Dmax, 3)))
    axbig.annotate("T = "+str(round(T, 3)), xy=(T, 0.01))
    axbig.annotate("Dmax", xy=(T, (F1_C1_T+F1_C0_T)/2))
    axbig.title.set_text('CDF of posterior probabilities (f1)')
    axbig.legend()
    axbig.set_xlim([0, 1])
    axbig.set_xlabel('P(C1|x)')
    axbig.grid(True)
    fig.savefig(os.path.join(save_dir, 'sample_distributions.png'), dpi=900)

def visualizeTSNE(Xscaled, number_of_ref, overlapped_index):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    embedding = manifold.TSNE(n_components=2)
    X_transformed = embedding.fit_transform(Xscaled)
    ax.scatter(X_transformed[:number_of_ref, 0], X_transformed[:number_of_ref, 1],marker='x', label='C1 reference set')
    ax.scatter(X_transformed[number_of_ref:, 0], X_transformed[number_of_ref:, 1], marker='^',label='C0 mixed set')
    ax.scatter(X_transformed[number_of_ref:750, 0], X_transformed[number_of_ref:750, 1],marker='^',color='r',label='ground truth malwares in C1')
    ax.scatter(X_transformed[overlapped_index, 0], X_transformed[overlapped_index, 1],s=90,facecolors='none', edgecolors='r',label='detected by QSL')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Detected Malwares')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    fig.savefig('feature_visualize.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)

def evaluate_results(Y_predicted,Y_actual):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_actual, Y_predicted)
    cm_df = pd.DataFrame(data=cm,index=["Actual Negatives", "Actual Positives"], columns=["Predicted Negatives", "Predicted Positives"])
    accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
    recall=cm[1,1]/(cm[1,0]+cm[1,1]) 
    precision=cm[1,1]/(cm[0,1]+cm[1,1])
    f1 = 2*precision*recall/(precision+recall)
    
    evaluation_results = {"confusion_matrix": cm_df,
                                  "accuracy": accuracy,
                                  "recall":recall,
                                  "precision": precision,
                                  "f1Score":f1}
    
    return evaluation_results

def print_results(evaluation_results):
    print(evaluation_results['confusion_matrix'])
    print("")
    print("Accuracy=",evaluation_results['accuracy'])
    print("Recall=", round(evaluation_results['recall'],4))
    print("Precision=",round(evaluation_results['precision'],4))
    print("F1 Score=",round(evaluation_results['f1Score'],4))


