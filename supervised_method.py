from scipy.io import loadmat
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking,construct_label_matrix_pan
from skfeature.function.statistical_based import f_score
from skfeature.function.sparse_learning_based import ll_l21
import datetime
import time
import numpy as np
from sklearn import preprocessing
from lassonet import LassoNetClassifier
import matplotlib.pyplot as plt
import random

seed = 1
random.seed(seed)
np.random.seed(seed)



def load_data(name):
    if name == "coil20":
        mat = loadmat('data/COIL20.mat')
        X = mat['X']  # data
        X = X.astype(float)
        y = mat['Y']  # label
        y = y[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    elif name == "mac":
        mat = loadmat('data/PCMAC.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    elif name=="SMK":
        mat = loadmat('data/SMK-CAN-187.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    elif name == "isolet":

        mat = loadmat('data/isolet.mat')
        X = mat['X']  # data
        # X = X.astype(float)
        y = mat['Y']  # label
        y = y[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    elif name == "MNIST":

        # mat = loadmat('data/mnist-original.mat')
        # X = mat['data']  # data
        # y = mat['label']  # label
        # y = y[:, 0]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')


    elif name == "har":
        X_train = np.loadtxt('data/UCI HAR Dataset/train/X_train.txt')
        y_train = np.loadtxt('data/UCI HAR Dataset/train/y_train.txt')
        X_test = np.loadtxt('data/UCI HAR Dataset/test/X_test.txt')
        y_test = np.loadtxt('data/UCI HAR Dataset/test/y_test.txt')


    return X_train, X_test, y_train, y_test

def select_features(method, X, y):

    if method =='fisher_score':
        score = fisher_score.fisher_score(X, y)
        idx = fisher_score.feature_ranking(score)
    elif method =='CIFE':
        # obtain the index of each feature on the training set
        # idx, _, _ = CIFE.cife(X, y, n_selected_features=k)
        idx, _, _ = CIFE.cife(X, y)
    elif method =='ICAP':
        # idx, _, _ = ICAP.icap(X, y, n_selected_features=k)
        idx, _, _ = ICAP.icap(X, y)
    elif method =='f_score':
        score = f_score.f_score(X, y)
        # rank features in descending order according to score
        idx = f_score.feature_ranking(score)
    elif method == 'RFS':
        # obtain the feature weight matrix
        y = construct_label_matrix(y)
        Weight = RFS.rfs(X, y, gamma=0.1)
        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)
    elif method == 'll_l21':
        Y = construct_label_matrix_pan(y)
        Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X, Y, 0.1, verbose=False)
        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)
    elif method =='lassonet':
        model = LassoNetClassifier(verbose=True)
        model.fit(X, y)
        score=(model.feature_importances_.numpy())
        idx = np.argsort(score,0)
        idx = idx[::-1]


    return idx

def main():
    # methods = ['fisher_score','ll_l21','f_score','RFS','ICAP','CIFE']
    methods = ['lassonet', 'fisher_score', 'll_l21',]
    # set the methods
    # methods = ['RFS']
    evaluate_methods=['KNN','ET','SVC']
    # set the dataset
    # dataset = ['isolet', 'har','coil20'] #without MNIST
    dataset = ['isolet', 'har','coil20','MNIST'] #MNIST
    # dataset = ['MNIST']
    # selected_list = [10, 20, 30, 40, 50, 75, 100, 125,150, 175, 200]

    # set the feature number
    selected_list = [25,50,75,100,150,200]
    color_list=['b','g','r','c','m','y','k']

    filename = "results_"
    file1 = open(filename + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".txt", "a")
    for name in dataset:
        KNNacc = []
        ETacc = []
        SVCacc = []
        baseline = []
        #load data and scaler
        X_train, X_test, y_train, y_test = load_data(name)
        indices=np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test[indices]


        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        for i in range(len(methods)):

            start = time.time()



            KNNacc.append([])
            ETacc.append([])
            SVCacc.append([])

            #create baseline use all features
            clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=1)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            baseline.append((accuracy_score(y_test, y_predict)))

            clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            baseline.append((accuracy_score(y_test, y_predict)))

            clf = svm.SVC()
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            baseline.append((accuracy_score(y_test, y_predict)))
            # run 3 times to get var and mean
            for number in range(3):
                idx = select_features(methods[i], X_train, y_train)
                KNNacc[i].append([])
                ETacc[i].append([])
                SVCacc[i].append([])
                for k in selected_list:
                    # idx = select_features(methods[i], X_train, y_train, k)
                    selected_features_train = X_train[:, idx[0:k]]
                    selected_features_test = X_test[:, idx[0:k]]

                    clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=1)
                    clf.fit(selected_features_train, y_train)
                    y_predict = clf.predict(selected_features_test)
                    KNNacc[i][number].append(float(accuracy_score(y_test, y_predict)))

                    clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
                    clf.fit(selected_features_train, y_train)
                    y_predict = clf.predict(selected_features_test)
                    ETacc[i][number].append(float(accuracy_score(y_test, y_predict)))

                    clf = svm.SVC()
                    clf.fit(selected_features_train, y_train)
                    y_predict = clf.predict(selected_features_test)
                    SVCacc[i][number].append(float(accuracy_score(y_test, y_predict)))

            end = time.time()
            print(methods[i],name,str(end-start))
        acc_set=[KNNacc,ETacc,SVCacc]
        # draw picture
        for i in range(len(evaluate_methods)):

            for j in range(len(methods)):
                # draw picture
                mm=np.mean(acc_set[i][j],axis=0)
                vv = np.var(acc_set[i][j], axis=0)
                plt.plot(selected_list, mm, color=color_list[j], label=methods[j], marker='o',linestyle='--')
                plt.fill_between(selected_list,mm-vv,mm+vv,alpha=0.3,facecolor=color_list[j])
            plt.axhline(baseline[i], label='full features')
            plt.axis([0, 200, 0.1, 1.0])
            plt.xlabel('Features selected')
            plt.ylabel(evaluate_methods[i]+' accuracy')
            plt.title(name)
            plt.legend()
            plt.savefig('picture/' + name + '_' + evaluate_methods[i] + '_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.png')
            plt.show()
        #write result to txt
        file1.write(name+'\n')
        for j in range(len(methods)):
            file1.write('method: '+methods[j]+'\n')
            print('method: ', methods[j])
            for k in range(len(selected_list)):
                file1.write('select number: '+ str(selected_list[k])+'\n')
                print('select number: ', selected_list[k])
                for i in range(len(evaluate_methods)):
                    mm = np.mean(acc_set[i][j], axis=0)
                    file1.write(evaluate_methods[i]+'acc = '+str('{:.3f}'.format(mm[k])) +', ')
                    print(evaluate_methods[i],'acc = ','{:.3f}'.format(mm[k]))
                file1.write('\n')

    file1.close()




if __name__ == '__main__':
    main()
