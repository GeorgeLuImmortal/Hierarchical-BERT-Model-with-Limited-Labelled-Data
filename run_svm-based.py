
# coding: utf-8

# In[ ]:


import numpy as np 
np.random.seed(2019)

from numpy import genfromtxt

import random as r
r.seed(2019)

import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

import os
os.environ['PYTHONHASHSEED'] = str(2019)


from sklearn.model_selection import train_test_split
from util import d, here

import pandas as pd
from argparse import ArgumentParser

import random, tqdm, sys, math, gzip
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

import gc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit,StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


import time

import shutil
from sklearn import metrics
from sklearn.utils import shuffle
from optparse import OptionParser


args = {

    ## directory for encoded embeddings
    'data_dir': 'datasets/',

    ## output of the experiments
    'output_dir': 'outputs/'
}

def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

class Normalize(object):
    
    def normalize(self, X_train, X_val):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
       
        return (X_train, X_val) 
    
    def inverse(self, X_train, X_val):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
    
        return (X_train, X_val) 


if __name__ == "__main__":



    parser = OptionParser(usage='usage: -r random_seeds -d dataset_name -t text_rep (options are fasttext or roberta-base) -s training_size')
    parser.add_option("-t","--text_rep", action="store", type="string", dest="text_rep", help="text representation used", default = 'fasttext')
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of data encoded by fasttext/roberta-base", default = 'longer_moviereview')
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['1988','1989'])
    parser.add_option('-s', '--training_size', type='string', action='callback',dest='training_size',callback=list_callback,default=['50','100'])

    (options, _ ) = parser.parse_args()

    

    if options.text_rep not in ['roberta-base', 'fasttext']:
        parser.error( "Must specify one text representation used, options are roberta-base or fasttext." )

    for number in options.training_size:
        if int(number)>200:
            parser.error( "The largest training size is 200, you can customize the maximum training size by modifying the corrsponding codes of initializing training set." )



   


    text_rep = options.text_rep
    train_sizes = [int(number) for number in options.training_size]
    dataset = options.dataset_name
    random_states = [int(number) for number in options.random_seeds]

    print('dataset name: ', dataset)
    print('initial random states: ', random_states)
    print('training set sizes: ', train_sizes)

    dir_neg = dataset + '_neg.csv'
    dir_pos = dataset + '_pos.csv'


    representations_neg = genfromtxt(os.path.join(args['data_dir'],'%s_data/'%(text_rep)+dir_neg), delimiter=',')
    representations_pos = genfromtxt(os.path.join(args['data_dir'],'%s_data/'%(text_rep)+dir_pos), delimiter=',')


    ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)
    labels = np.array([0]*len(representations_neg)+[1]*len(representations_pos))


    seeds =  [i for i in range(1988,1993)]

    dataset_len = len(ulti_representations)

    c_weight = 'balanced'

    for train_size in train_sizes:
        df = pd.DataFrame()
        df_auc = pd.DataFrame()
        matrices = []
        aucs = []
        for seed in random_states:
            

        #     X_train, X_test, y_train, y_test = train_test_split(
        #     ulti_representations, labels, test_size=test_size, random_state=seed,stratify=labels)
            index_shuffle = shuffle([i for i in range(len(ulti_representations))], random_state=seed)

            total_train_shuffle = index_shuffle[:200]
            train_shuffle = total_train_shuffle[:train_size]
            test_shuffle = index_shuffle[200:]

            X_train, X_test = ulti_representations[train_shuffle],ulti_representations[test_shuffle]
            y_train, y_test = labels[train_shuffle],labels[test_shuffle]

            normalizer = Normalize()
            X_train, X_val = normalizer.normalize(X_train, X_test) 



            print('start gridsearch ...')
            parameters = [
                        {'kernel': ['linear'],
                         'C': [ 0.01, 0.1, 1,10]}]

            cv = StratifiedKFold(n_splits=5,random_state=seed)
            svc = SVC(probability=True,random_state=2019,max_iter=10000,class_weight = c_weight)
            classifier = GridSearchCV(svc, parameters, cv=cv,scoring='accuracy',n_jobs=8,verbose = 0)
            classifier.fit(X_train, y_train)
            print('best parameters is ', classifier.best_params_)
            best_params_ = classifier.best_params_


            kernel = best_params_['kernel']
            C = best_params_['C']

            classifier = SVC(probability=True,random_state=2019,C=C,kernel=kernel,max_iter=10000,class_weight = c_weight)
            classifier.fit(X_train, y_train)

            y_pred_prob = classifier.predict_proba(X_val)
            y_eval_prob_pos = np.array(y_pred_prob)[:,1]
            y_pred = np.argmax(y_pred_prob,axis=1)
            y_true = y_test

            acc = accuracy_score(y_true, y_pred)
            f_score = f1_score(y_true,y_pred,average='macro')
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_eval_prob_pos,pos_label=1)
            aucs.append(metrics.auc(fpr, tpr))


            print(classification_report(y_true,y_pred),acc,f_score,metrics.auc(fpr, tpr))
            print('TP:',tp,'TN:',tn,'FP:',fp,'FN:',fn)

            raw_dict = {'TP':tp,'TN':tn,'FP':fp,'FN':fn}
            matrices.append(raw_dict)

        df['result'] = [row for row in matrices]
        df['seed'] = random_states
        df = df.set_index('seed')
        df.to_csv(os.path.join(args['output_dir'],'raw_%s_%s_SVM_%s.csv'%(dataset,text_rep,train_size)),index=True)

        df_auc['result'] = [row for row in aucs]
        df_auc['seed'] = random_states
        df_auc = df_auc.set_index('seed')
        df_auc.to_csv(os.path.join(args['output_dir'],'auc_%s_%s_SVM_%s.csv'%(dataset,text_rep,train_size)),index=True)


