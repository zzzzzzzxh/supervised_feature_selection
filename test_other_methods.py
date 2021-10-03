#!/usr/bin/env python3
# ## Supervised Methods
# ##### RFS 
# 
# ### Unsupervised Methods
# #### sparse learning based
# ##### 1. UDFS
# ##### 2. MCFS
# #### similarity based
# ##### 3. Lap Score
# 
# ## Feature Extrction
# ##### PCA
# 

# In[1]:

# IMPORTS
import resource
import scipy
from scipy.io import loadmat
#print("Memory usage0: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from skfeature.utility import construct_W
#print("Memory usage1: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
from skfeature.utility import unsupervised_evaluation
#print("Memory usage2: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.function.sparse_learning_based import MCFS
#from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.similarity_based import lap_score

#print("Memory usage3: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

#from sklearn.decomposition import PCA
#from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
#print("Memory usage30: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#from IPython.display import clear_output
from sklearn import preprocessing
#print("Memory usage31: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
import os; import sys; sys.path.append(os.getcwd())
#print("Memory usage32: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
from utils import check_path
import datetime
# ## Utilities
#print("Memory usage4: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
import time

import argparse


#import scipy.io as spio

def load_data(name, noise_dim_pro = 1):
    artificial = ["artificial_NS_10000_NF_10000_0", "artificial_NS_2000_NF_200_0", \
            "artificial_NS_30000_NF_10000_0", "artificial_NS_40000_NF_10000_0",\
            "artificial_NS_40000_NF_5000_0", "artificial_NS_40000_NF_6000_0",\
             "artificial_NS_40000_NF_8000_0", "artificial_NS_40000_NF_8000_1"]
    if name == "coil20":
        mat = scipy.io.loadmat('../QuickSelection/datasets/COIL20.mat')
        X = mat['fea']
        y = mat['gnd'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        
    elif name=="madelon":
        import urllib.request as urllib2
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        X_train = np.loadtxt(urllib2.urlopen(train_data_url))
        y_train = np.loadtxt(urllib2.urlopen(train_resp_url))
        X_test =  np.loadtxt(urllib2.urlopen(val_data_url))
        y_test =  np.loadtxt(urllib2.urlopen(val_resp_url))
        
    elif name=="isolet":
        import pandas as pd
        data= pd.read_csv('../QuickSelection/datasets/isolet.csv')
 
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
        y_train = y_train -1
        y_test = y_test - 1
        #y_train = y_train.astype("float")
        #y_test = y_test.astype("float")

        
    elif name=="har":         
        X_train = np.loadtxt('../QuickSelection/datasets/UCI HAR Dataset/train/X_train.txt')
        y_train = np.loadtxt('../QuickSelection/datasets/UCI HAR Dataset/train/y_train.txt')
        X_test =  np.loadtxt('../QuickSelection/datasets/UCI HAR Dataset/test/X_test.txt')
        y_test =  np.loadtxt('../QuickSelection/datasets/UCI HAR Dataset/test/y_test.txt')
       
    elif name == "MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')    
    
    elif name == "MNIST2":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32') 
        print(X_train.shape)
        for i in range(X_train.shape[1] * noise_dim_pro):           
            s  = np.random.normal(0, 1, X_train.shape[0])  
            s = np.reshape(s, (s.shape[0],1))
            X_train = np.concatenate((X_train, s), axis=1) 
            s  = np.random.normal(0, 1, X_test.shape[0])  
            s = np.reshape(s, (s.shape[0],1))            
            X_test = np.concatenate((X_test, s), axis=1) 
        print(X_train.shape)    
        
    elif name=="SMK":
        mat = loadmat('./data/SMK-CAN-187.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
    
    elif name=="GLA":
        mat = loadmat('./data/GLA-BRA-180.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)     
    elif name=="mac":
        mat = loadmat('./data/PCMAC.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)     
    elif name=="news":
        X_train = np.load('../QuickSelection/datasets/news_X_train.npy')
        X_test  = np.load('../QuickSelection/datasets/news_X_test.npy')
        y_train = np.load('../QuickSelection/datasets/news_y_train.npy')
        y_test  = np.load('../QuickSelection/datasets/news_y_test.npy') 
    elif name in artificial:
        X_train = np.load('../QuickSelection/datasets/X_train_'+name+".npy")
        X_test  = np.load('../QuickSelection/datasets/X_test_'+name+".npy")
        y_train = np.load('../QuickSelection/datasets/y_train_'+name+".npy")
        y_test  = np.load('../QuickSelection/datasets/y_test_'+name+".npy")    
        
        
    
    
    elif name =="news2":
        """
        from sklearn.datasets import fetch_20newsgroups
        groups = fetch_20newsgroups()
        import nltk
        nltk.download('all')
        #nltk.download()
        #nltk.download('names')
        data_train = fetch_20newsgroups(subset='train', random_state=42)
        train_label = data_train.target
        data_test = fetch_20newsgroups(subset='test', random_state=42)
        test_label = data_test.target
        from collections import defaultdict
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import names

        all_names = names.words()
        WNL = WordNetLemmatizer()
        def clean(data):
            cleaned = defaultdict(list)
            count = 0
            for group in data:
                for words in group.split():
                    if words.isalpha() and words not in all_names:
                        cleaned[count].append(WNL.lemmatize(words.lower()))
                cleaned[count] = ' '.join(cleaned[count])
                count +=1 
            return(list(cleaned.values()))
        
        x_train = clean(data_train.data)        
        from sklearn.feature_extraction.text import TfidfVectorizer

        tf = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = tf.fit_transform(x_train)
        X_train = X_train.toarray()
        
        x_test = clean(data_test.data)
        X_test = tf.transform(x_test)
        X_test = X_test.toarray()
        
        y_train = train_label
        y_test = test_label
        y_test = test_label
        """
        X_train = np.load('../QuickSelection/datasets/news2_X_train.npy')
        X_test  = np.load('../QuickSelection/datasets/news2_X_test.npy')
        y_train = np.load('../QuickSelection/datasets/news2_y_train.npy')
        y_test  = np.load('../QuickSelection/datasets/news2_y_test.npy')  
        
    print(X_train.shape)
    print(y_train.shape)
    print("doneeeeee", flush=True)
    
    
    #scaler = preprocessing.StandardScaler().fit(X_train)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    return X_train, y_train, X_test, y_test
    
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from skfeature.utility import unsupervised_evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def eval_subset(train, test):
    n_clusters = len(np.unique(train[2]))
    
    clf = RandomForestClassifier(n_estimators = 50, n_jobs = -1)
    clf.fit(train[0], train[2])
    DTacc = float(clf.score(test[0], test[2]))
    print(DTacc)
    
    
    clf = KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute', n_jobs = 1)
    clf.fit(train[0], train[2])
    acc = float(clf.score(test[0], test[2]))
    
    #LR = LinearRegression(n_jobs = -1)
    #LR.fit(train[0], train[1])
    MSELR = 0#float(((LR.predict(test[0]) - test[1]) ** 2).mean())
    
    MSE = 0 #float((((decoder((train[0], train[1]), (test[0], test[1])) - test[1]) ** 2).mean()))
    
    max_iters = 10
    cnmi, cacc = 0.0, 0.0
    #for iter in range(max_iters):
    #    nmi, acc = unsupervised_evaluation.evaluation(train[0], n_clusters = n_clusters, y = train[2])
    #    cnmi += nmi / max_iters
    #    cacc += acc / max_iters
    #from sklearn.metrics.cluster import normalized_mutual_info_score
    #normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
    
    print('nmi = {:.3f}, acc = {:.3f}'.format(cnmi, cacc))
    print('acc = {:.3f}, DTacc = {:.3f}, MSELR = {:.3f}, MSE = {:.3f}'.format(acc, DTacc, MSELR, MSE))
    return MSELR, MSE, acc, DTacc, float(cnmi), float(cacc)

def select_features(method, X, y, k):
    if method =='MCFS':
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs)           
        Weight = MCFS.mcfs(X, n_selected_features=k, W=W, n_clusters=10)
        # sort the feature scores in an ascending order according to the feature scores
        return MCFS.feature_ranking(Weight)
        
    elif method == 'lap_score': 
        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        # obtain the scores of features
        score = lap_score.lap_score(X, W=W)
        return lap_score.feature_ranking(score)
    
    elif method == 'UDFS':
        Weight = UDFS.udfs(X, gamma=0.1, n_clusters=10)
        # sort the feature scores in an ascending order according to the feature scores
        return feature_ranking(Weight) 
    
    elif method == 'RFS':
        Weight = RFS.rfs(X, construct_label_matrix(y), gamma=0.1)
        # sort the feature scores in an ascending order according to the feature scores
        return feature_ranking(Weight)
        
print("Memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)





parser = argparse.ArgumentParser(description='Test other methods')
parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
args = parser.parse_args()
dataset = args.dataset_name 
filename = "./results/"
check_path(filename)
file1 = open(filename+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+"_"+dataset+".txt","a") 

file1.write("------------------------------------------------------------------")
file1.write("\nDataset = "+ dataset)
X_train, y_train, X_test, y_test = load_data(dataset)
file1.write("\nX_train shape = "+ str(X_train.shape))
file1.write("\nX_test shape = " + str(X_test.shape ))
file1.write("\ny_train shape = "+ str(y_train.shape))
file1.write("\ny_test shape = " + str(y_test.shape ))    



results_acc = []
results_acc2 = []
results_nmi = []
result_nmi = [["NMI"]]
result_acc = [["ACC"]]
result_acc2 = [["ACC2"]]
methods = ['MCFS', 'lap_score', 'UDFS']
methods = ['lap_score']
for method in methods:
    result_nmi.append([method])
    result_acc.append([method])
    result_acc2.append([method])


for j in range(1):
    file1.write("\n-----------------------------------------------------------------------")
    file1.write("\niter {}:".format(j))
    
    col = 0
    result_nmi[col].append(dataset)
    result_acc[col].append(dataset)
    result_acc2[col].append(dataset)
    k = 50
    #if dataset == 'madelon':
    #    k = 20
    for method in methods:
        start = time.time() 
        col += 1
       
        idx = select_features(method, X_train, y_train, k)      
        print("data: ", dataset, flush=True)
        print("k=", k)
        print("Memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        
        # obtain the dataset on the selected features
        X_train2 = X_train[:, idx[:k]]
        X_test2  = X_test[:, idx[:k]]
        end = time.time()
        file1.write("\n"+method+"\telapsed time : "+ str( end - start))
        """#---------------------------------------------
        from sklearn.metrics.cluster import normalized_mutual_info_score
        from sklearn.cluster import KMeans
        k_means = KMeans(n_clusters=10)
        k_means.fit(X_train2)
        y_predict = k_means.labels_
        nmi = normalized_mutual_info_score(y_train.ravel(), y_predict.ravel())
        file1.write("\nnmi {}:".format(str(nmi)))
        """
        #---------------------------------------------
        
        y_test  = y_test.ravel().astype("int")
        y_train = y_train.ravel().astype("int")
        MSELR, MSE, acc, DTacc, cnmi, cacc =   eval_subset([X_train2, X_train, y_train],
                        [X_test2,  X_test,  y_test])

        result_nmi[col].append(round(cnmi,4))
        result_acc[col].append(round(cacc,4))
        result_acc2[col].append(round(DTacc,4))

results_nmi.append(result_nmi)
results_acc.append(result_acc)
results_acc2.append(result_acc2)
file1.write("\n\n\n")
print("done", flush=True)    
from texttable import Texttable






table = []
table.append(["NMI"])
for d in range(len(results_nmi)):
    result_nmi = results_nmi[d]
    table[0].append(result_nmi[0][1])
    num_methods = len(result_nmi)-1
    num_runs = len(result_nmi[0])-1
    metrics = np.zeros((num_methods,num_runs))
    for i in range(num_methods):
        if d == 0:
            table.append([result_nmi[i+1][0]])
        for j in range(num_runs):
            metrics[i,j] = result_nmi[i+1][j+1]

    mean = np.mean(metrics, axis = 1)
    std = np.std(metrics, axis = 1)
    for i in range(len(mean)):
        table[i+1].append("{0:.4f}".format(round(mean[i],4))+""+u"\u00B1"+"{0:.4f}".format(round(std[i],4)))
    t = Texttable()
    t.add_rows(table)
    file1.write(t.draw())

file1.write("\n\n\n")
table = []
table.append(["ACC"])
for d in range(len(results_acc)):
    result_acc = results_acc[d]
    table[0].append(result_acc[0][1])
    num_methods = len(result_acc)-1
    num_runs = len(result_acc[0])-1
    metrics = np.zeros((num_methods,num_runs))
    for i in range(num_methods):
        if d == 0:
            table.append([result_acc[i+1][0]])
        for j in range(num_runs):
            metrics[i,j] = result_acc[i+1][j+1]

    mean = np.mean(metrics, axis = 1)
    std = np.std(metrics, axis = 1)
    for i in range(len(mean)):
        table[i+1].append("{0:.4f}".format(round(mean[i],4))+""+u"\u00B1"+"{0:.4f}".format(round(std[i],4)))
    t = Texttable()
    t.add_rows(table)
    file1.write(t.draw())

file1.write("\n\n\n")
table = []
table.append(["ACC2"])
for d in range(len(results_acc2)):
    result_acc2 = results_acc2[d]
    table[0].append(result_acc2[0][1])
    num_methods = len(result_acc2)-1
    num_runs = len(result_acc2[0])-1
    metrics = np.zeros((num_methods,num_runs))
    for i in range(num_methods):
        if d == 0:
            table.append([result_acc2[i+1][0]])
        for j in range(num_runs):
            metrics[i,j] = result_acc2[i+1][j+1]

    mean = np.mean(metrics, axis = 1)
    std = np.std(metrics, axis = 1)
    for i in range(len(mean)):
        table[i+1].append("{0:.4f}".format(round(mean[i],4))+""+u"\u00B1"+"{0:.4f}".format(round(std[i],4)))
    t = Texttable()
    t.add_rows(table)
    file1.write(t.draw())
file1.close()    





