# imports

"""*************************************************************************"""
"""                           IMPORT LIBRARIES                              """
  
import numpy as np 
import pandas as pd
import urllib.request as urllib2 
import errno
#import tensorflow as tf
import argparse
import scipy
from scipy.io import loadmat
# sklearn
import  sklearn  
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix


import  sklearn
import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from skfeature.utility import unsupervised_evaluation
from sklearn.neighbors import KNeighborsClassifier
import sklearn 
from sklearn import svm
import numpy as np
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


import numpy as np
import os; import sys; sys.path.append(os.getcwd())
import datetime
import time
import sklearn
import scipy
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score 


"""*************************************************************************"""
"""                              Functions                                  """

def check_path(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise 

import pickle
def save_obj(name, obj):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)





def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('--dataset_name', type=str, default='coil20', help='dataset name')
    parser.add_argument("--K", help="K", type=int , required=False)
    parser.add_argument("--Kp", type=float, default=0.1, help="fraction of features to be selected", required=False)
    parser.add_argument("--exp", type=int , required=True, help="Experiment ID")
    parser.add_argument('--exp_extra', type=str, default="", help='extra')
    
    # Model settings
    parser.add_argument("--epsilon", type=int, default=13, help="epsilon")
    parser.add_argument("--zeta_in", type=float, default=0.4, help="zeta input")
    parser.add_argument("--zeta_hid", type=float, default=0.2, help="zeta hidden")
    parser.add_argument('--model', type=str, required=True, help='model')
    parser.add_argument("--num_hidden", default=1000, help="num_hidden", type=int)
    parser.add_argument("--update_interval", default=1, help="update_interval", type=int)
    

    # Train options
    parser.add_argument("--epochs", help="epochs", type=int)
    #parser.add_argument('--rounds', nargs="+", type=int)
    parser.add_argument("--seed", default=0, help="seed", type=int)
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples per mini-batch')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0, help='dropout rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    

    args = parser.parse_args()
    return args





"""*************************************************************************"""
"""                             Load data                                   """
    
def load_data(args):
    import numpy as np
    name = args.dataset_name
    import numpy as np
    import pandas as pd
    ###########################################################################################
    ########### Image
    if name == "coil20":
        mat = scipy.io.loadmat('../0datasets/COIL20.mat')
        X = mat['fea']
        y = mat['gnd'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        y_test = y_test.ravel()
        y_train = y_train.ravel()
        
    if name == "USPS":
        mat = scipy.io.loadmat('../0datasets/USPS.mat')
        X = mat['fea']
        y = mat['gnd'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        y_test = y_test.ravel()
        y_train = y_train.ravel()
    
    elif name == "MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')  
        
    elif name == "Fashion-MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')    
    
    
    ###########################################################################################
    ########### Tabular
    elif name=="isolet":
        import pandas as pd 
        data= pd.read_csv('../0datasets/isolet.csv')
        data = data.values 
        X = data[:,:-1]
        X = X.astype("float")
        y = data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        for i in range(len(y_train)):
            if len(y_train[i])==4:
                y_train[i] = int(y_train[i][1])*10 + int(y_train[i][2])
            elif len(y_train[i])==3:
                y_train[i] = int(y_train[i][1])
        for i in range(len(y_test)):
            if len(y_test[i])==4:
                y_test[i] = int(y_test[i][1])*10 + int(y_test[i][2])
            elif len(y_test[i])==3:
                y_test[i] = int(y_test[i][1])
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')


    elif name=="madelon":
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        X_train = np.loadtxt(urllib2.urlopen(train_data_url))
        y_train = np.loadtxt(urllib2.urlopen(train_resp_url))
        X_test =  np.loadtxt(urllib2.urlopen(val_data_url))
        y_test =  np.loadtxt(urllib2.urlopen(val_resp_url))
        y_train[y_train < 0] = 0
        y_test[y_test < 0] = 0
        
    
    ###########################################################################################
    ########### Time Series    
    elif name=="har":         
        X_train = np.loadtxt('../0datasets/UCI HAR Dataset/train/X_train.txt')
        y_train = np.loadtxt('../0datasets/UCI HAR Dataset/train/y_train.txt')
        X_test =  np.loadtxt('../0datasets/UCI HAR Dataset/test/X_test.txt')
        y_test =  np.loadtxt('../0datasets/UCI HAR Dataset/test/y_test.txt')
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')
    

    ###########################################################################################
    ########### Biological   
    
    elif name=="SMK":
        mat = scipy.io.loadmat('./datasets/SMK_CAN_187.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    elif name == "Carcinom":
        mat = scipy.io.loadmat('./datasets/Carcinom.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
    elif name == "lung":
        mat = scipy.io.loadmat('./datasets/lung.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)    
        

    elif name == "synthetic_100":
        from sklearn.datasets import make_classification
        # Define the number of features for each type
        n_features = 100
        n_informative = 20
        n_redundant = 40
        n_repeated = 0
        n_useless = 40

        # Create Labels
        informative_labels = [f'Informative {ii}' for ii in range(1, n_informative + 1)]
        redundant_labels = [f'Redundant {ii}' for ii in range(n_informative + 1, n_informative + n_redundant + 1)]
        repeated_labels = [f'Repeated {ii}' for ii in range(n_informative + n_redundant+ 1, n_informative + n_redundant + n_repeated + 1)]
        useless_labels = [f'Useless {ii}' for ii in range(n_informative + n_redundant + n_repeated + 1, n_features + 1)]
        labels = informative_labels + redundant_labels + repeated_labels + useless_labels

        # Get data
        X, y = make_classification(n_samples = 2000, n_features = n_features,
                                   n_informative = n_informative,
                                   n_redundant = n_redundant , n_repeated = n_repeated,
                                   n_clusters_per_class = 2, class_sep = 0.5, flip_y = 0.05,
                                   random_state = 42, shuffle = False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)    
        print("train labels: ", np.unique(y_train))
        print("test labels: ", np.unique(y_test))
        args.labels = labels

    
    
    #
    if name == "madelon":
        #scaler = preprocessing.StandardScaler().fit(X_train)
        X = preprocessing.StandardScaler().fit_transform(np.concatenate((X_train, X_test)))
        X_train = X[: y_train.shape[0]]
        X_test = X[y_train.shape[0]:]
    else:
        X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((X_train, X_test)))
        X_train = X[: y_train.shape[0]]
        X_test = X[y_train.shape[0]:]
    
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test) 
    #noClasses = int(np.max(y_train)+1)
    #print("noClasses ", noClasses)

    if name in ["har", "isolet", "Carcinom", "TOX_171", "epileptic"]:
        y_train = y_train - 1
        y_test = y_test - 1
    print("train labels: ", np.unique(y_train))
    print("test labels: ", np.unique(y_test))
    
    args.num_classes = len(np.unique(y_train))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32') 
    y_train = y_train.astype('int')
    y_test  = y_test.astype('int')
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)     
    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))
    return   args, [X_train,X_test,y_train,y_test]

    






"""*************************************************************************"""
"""                             Evaluation                                  """



def eval_subset_supervised(train, test):
    print("X_train shape = "+ str( train[0].shape))
    print("X_test shape = "+ str( test[0].shape))

    clf = KNeighborsClassifier(n_neighbors = 1, algorithm = 'brute', n_jobs = 1)
    clf.fit(train[0], train[2])
    KNNacc = float(clf.score(test[0], test[2]))
    
    
    clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1)
    clf.fit(train[0], train[2])
    ETacc = float(clf.score(test[0], test[2]))
    
    
    
    clf = svm.SVC()
    clf.fit(train[0], train[2])
    SVCacc = float(clf.score(test[0], test[2]))

  
    print('KNNacc = {:.3f}, ETacc = {:.3f}, SVCacc = {:.3f}'.format(KNNacc, ETacc, SVCacc))

    return KNNacc, ETacc, SVCacc
    
    
    
 
def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)



from sklearn.cluster import KMeans
def evaluation(X_selected, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results
    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels
    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy
    """
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(X_selected)
    y_predict = k_means.labels_

    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return acc, nmi
    


from sklearn.ensemble import ExtraTreesClassifier
def eval_subset_unsupervised(train, test):
    n_clusters = len(np.unique(train[2]))

    max_iters = 5
    cacc = 0.0
    cnmi = 0.0
    for iter in range(max_iters):
        acc, nmi = evaluation(train[0], n_clusters = n_clusters, y = train[2])
        cacc += acc / max_iters
        cnmi += nmi / max_iters
    
    print('CluACC = {:.3f}, NMI = {:.3f}'.format(cacc, cnmi))
    
        
    return  cacc, cnmi
        
    














"""    
    elif name=="GLA":
        mat = scipy.io.loadmat('./datasets/GLA-BRA-180.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.StandardScaler().fit(X_train)
    
    elif name == "leukemia":
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import KFold
        print("Loading data...")
        dataset = fetch_openml("leukemia")
        X = np.asfortranarray(dataset.data.astype(float))
        y = 2 * ((dataset.target == "AML") - 0.5)
        y[y<0] = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    elif name == "TOX_171":
        mat = scipy.io.loadmat('./datasets/TOX_171.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    elif name == "epileptic":
        import pandas as pd
        df=pd.read_csv('./datasets/data.csv')
        X=df.values
        X=X[:,1:-1]
        y=np.array(df['y'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    elif name == "micromass":
        with open("./datasets/micromass.csv") as train_file:
            lines = train_file.readlines()
            X = np.array([[float(i) for i in line.strip().split(",")[:-1]] for line in lines])
            y = np.array([int(line.strip().split(",")[-1]) - 1 for line in lines])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)"""