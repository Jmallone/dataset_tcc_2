# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy
from single_split_gridsearch import SingleSplitGridSearch
from sklearn.metrics import classification_report, accuracy_score, f1_score
import dill
import click
import sys
import glob
import itertools 
import pprint

def get_final_predictions(n_patches, predicted):
    songs = int(predicted.shape[0] / n_patches)
    r = predicted.reshape((songs,n_patches))
    #print r[0]
    r = [np.unique(i, return_counts=True) for i in r]
    r = [ i[0][np.argmax(i[1])] for i in r]
    
    return r

def get_final_predictions_sumvote(n_patches, predicted):
    songs = int(predicted.shape[0] / n_patches)
    r = predicted.reshape((songs, n_patches,-1))    
    return np.argmax(np.sum(r, 1), -1) + 1

def get_labels(n_patches, labels):
    o = []
    for i in range(0,len(labels),n_patches):
        o.append(labels[i])
    return np.array(o)

def load_data_shape(filename):
    with open(filename) as f:
        c = f.readline()
    c = [i for i in c.strip().split(" ")]
    return len(c)

def load_data(filename, class_idx, filename_idx, features_idxs):
    print (class_idx, filename_idx, features_idxs)
    with open(filename) as f:
        c = f.readlines()
    c = np.array([np.array(i.strip().split(" ")) for i in c])
    X = c[:,features_idxs].astype(float)
    Y = c[:,class_idx]
    F = c[:,filename_idx]
    
    return X, Y, F

#Essa função converte as features em formato libsvm (txt) para o formato do numpy
def libsvm_to_numpy(folds, patches, input_proto, output_proto):

    for fold, n_patches in itertools.product(folds, patches):
        fname = input_proto % (n_patches, fold)
        print (fname)
        
        X, Y, F = load_data(fname, 0, 1, slice(2,load_data_shape(fname)))
        Y = Y.astype(int)
        F = np.array(["%04d.bmp" % int(i) for i in F])
        
        X_out = (output_proto % (fold, n_patches)) + "_X.npy"
        Y_out = (output_proto % (fold, n_patches)) + "_Y.npy"
        F_out = (output_proto % (fold, n_patches)) + "_F.npy"
        
        print( "\t%s\n\t%s\n\t%s" % (X_out, Y_out, F_out))
        
        np.save(X_out, X)
        np.save(Y_out, Y)
        np.save(F_out, F)

def run_experiment(validation_test_size, folds, total_patches, proto_x, proto_y,
    exp_name, njobs):

    results = {}
        
    for patches in total_patches:
        print("patches: %d" % (patches))
        results[patches] = {'max_f1' : [], 'sum_f1' : [], 'max_acc' : [], 'sum_acc' : []}

        for test_fold in folds:
            print("fold: %d" % (test_fold))

            train_folds = sorted(list(set(folds) - set([test_fold])))
            
            train_X = np.concatenate([np.load(proto_x % (f,patches)) for f in train_folds])
            train_Y = np.concatenate([np.load(proto_y % (f,patches)) for f in train_folds])
            
            test_X = np.load(proto_x % (test_fold,patches))
            test_Y = np.load(proto_y % (test_fold,patches))

            ss = StandardScaler()
            train_X = ss.fit_transform(train_X)
            test_X = ss.transform(test_X)
            
            params = {'C' : [1, 10, 100, 1000], 'gamma' : ['auto', 2e-1, 2e-2, 2e-3]}
            svm = SVC(probability=True)

            clf = SingleSplitGridSearch(svm, params, n_jobs=njobs, verbose=50, test_size=validation_test_size)
            clf.fit(train_X, train_Y)

            print (clf.best_params_)

            predicted = clf.predict(test_X)
            prob = clf.best_estimator_.predict_proba(test_X)

            l = get_labels(patches, test_Y)
            p = get_final_predictions(patches, predicted)
            acc, f1 = accuracy_score(l, p), f1_score(l, p, average='weighted')
            results[patches]['max_acc'].append(acc)
            results[patches]['max_f1'].append(f1)
            print("MAX: acc: %.3f f1: %.3f" % (acc, f1))
            
            p = get_final_predictions_sumvote(patches, prob)
            acc, f1 = accuracy_score(l, p), f1_score(l, p, average='weighted')
            results[patches]['sum_acc'].append(acc)
            results[patches]['sum_f1'].append(f1)  
            print("SUM: acc: %.3f f1: %.3f" % (acc, f1))
            
            sys.stdout.flush()
                
    return results

if __name__ == "__main__":

    #Essa é a pasta que deve conter os arquivos de features em txt
    features_folder = "../Features/"

    folds = [1, 2, 3]
    patches = [1, 3, 5, 10]
    input_proto = features_folder + "Features_inceptionFeatures_LMD_%dx1_Fold%d_513x1599.txt"
    output_proto = features_folder + "LMD_Fold%d_Inception_%dx1"

    libsvm_to_numpy(folds, patches, input_proto, output_proto)

    results = run_experiment(0.2, 
        folds, 
        patches, 
        output_proto + "_X.npy", 
        output_proto + "_Y.npy",
        "lmd_inception",
        njobs=4)

    pprint.pprint(results)