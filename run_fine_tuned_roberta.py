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
import math

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

import fine_tuned_bert
import time

import shutil
from sklearn.utils import shuffle
import glob
import pickle
from optparse import OptionParser

args = {

    ## directory for original text data
    'text_data_dir:': 'raw_corpora/',
    ## directory for encoded embeddings
    'data_dir': 'datasets/roberta-base_data/',
    ## output of the experiments
    'output_dir': 'outputs/',
    ## detailed output of each step
    'verbose_output_dir': 'outputs/fine_tuned_results/',

    ## text data used for fine-tuning
    'intermediate_dir': 'fine_tuned_data/',
    'cuda_num': 1,
}


def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))





def initial_seed_dataset(n_initial, Y,random_state):
    
    np.random.seed(random_state)
    
    df = pd.DataFrame()
    df['label'] = Y

    Samplesize = n_initial  #number of samples that you want       
    initial_samples = df.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

    permutation = [index[1] for index in initial_samples.index.tolist()]
    
    print ('initial random chosen samples', permutation)
    
    return permutation





def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)





def read_data(path):

    corpora = []
    for filename in os.listdir(path):
        print(filename)

        df_temp = pd.read_csv(path+filename)

        corpora.append(df_temp.text.tolist())

    class_one_len = len(corpora[0])
    class_two_len = len(corpora[1])

    return corpora, class_one_len, class_two_len




def construct_train_set(dataset,permutation,permutation_test):
    print('constructing new text training set.....')
    corpora, class_one_len, class_two_len = read_data('./raw_corpora/'+dataset+'/')


    texts = np.array(corpora[0]+corpora[1])
    labels = np.array([0]*class_one_len+[1]*class_two_len)


    X_train, y_train = texts[permutation], labels[permutation]
    X_test, y_test = texts[permutation_test], labels[permutation_test]
    
    train_df = pd.DataFrame({'id':range(len(X_train)),'label':y_train,'alpha':['a']*len(X_train),'text':X_train})
    dev_df = pd.DataFrame({'id':range(len(X_test)),'label':y_test,'alpha':['a']*len(X_test),'text':X_test})
    
    print(len(train_df),len(dev_df))
    
    train_df.to_csv('./fine_tuned_data/train.tsv', sep='\t', index=False, header=False)
    dev_df.to_csv('./fine_tuned_data/dev.tsv', sep='\t', index=False, header=False)




if __name__ == "__main__":



    parser = OptionParser(usage='usage: -r random_seeds -d dataset_name -e no_epochs -t training_size' )
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of data encoded by roberta-base", default = 'longer_moviereview')
    parser.add_option("-e","--no_epochs", action="store", type="int", dest="no_epochs", help="the number of epochs for fine tuning",default=5)
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['1988','1989'])
    parser.add_option('-s', '--training_size', type='string', action='callback',dest='training_size',callback=list_callback,default=['50','100'])

    (options, _ ) = parser.parse_args()

    for number in options.training_size:
        if int(number)>200:
            parser.error( "The largest training size is 200, you can customize the maximum training size by modifying the corrsponding codes of initializing training set." )

    

    train_sizes = [int(number) for number in options.training_size]
    random_states = [int(number) for number in options.random_seeds]
    dataset = options.dataset_name
    no_epochs = options.no_epochs

    print('number of epochs: ', no_epochs)
    print('dataset name: ', dataset)
    print('initial random states: ', random_states)
    print('training set sizes: ', train_sizes)


    dir_neg = dataset + '_neg.csv'
    dir_pos = dataset + '_pos.csv'
    representations_neg = genfromtxt('./datasets/roberta-base_data/'+dir_neg, delimiter=',')
    representations_pos = genfromtxt('./datasets/roberta-base_data/'+dir_pos, delimiter=',')

    labels = np.array([0]*len(representations_neg)+[1]*len(representations_pos))



    for train_size in train_sizes:
        
        saving_steps = math.ceil(train_size/4)
        df = pd.DataFrame()
        df_auc = pd.DataFrame()
        matrices = []
        eval_aucs = []
        
        for seed in random_states:
            print('------start-----',seed)
            
            df_train_auc = pd.DataFrame()
            df_train_acc = pd.DataFrame()
            df_test_auc = pd.DataFrame()
            df_test_acc = pd.DataFrame()
            df_loss = pd.DataFrame()
            df_raw = pd.DataFrame()
            
            files = glob.glob('./fine_tuned_outputs/*')

            for f in files:
                shutil.rmtree(f)
            index_shuffle = shuffle([i for i in range(len(labels))], random_state=seed)

            total_train_shuffle = index_shuffle[:200]
            
            train_shuffle = total_train_shuffle[:train_size]
            
            test_shuffle = index_shuffle[200:]
            
            

            construct_train_set(dataset,train_shuffle,test_shuffle)

            checkpoint, accs, train_accs, aucs, train_aucs, tr_losses, eval_confusion_matrices = fine_tuned_bert.fine_tuned(dataset,'roberta-base',saving_steps,1e-5,no_epochs)

            df_train_acc[seed]=list(train_accs.values())
            df_train_auc[seed]=list(train_aucs.values())
            df_test_auc[seed]=list(aucs.values())
            df_test_acc[seed]=list(accs.values())
            df_loss[seed]=tr_losses
            df_raw[seed]=list(eval_confusion_matrices.values())
            

            ## Results of all fine tuned checkpoints of training set/testing set
            df_train_auc.to_csv('./outputs/fine_tuned_results/all_train_auc_%s_s%s_%s_fine_tuned.csv'%(dataset,seed,train_size))
            df_train_acc.to_csv('./outputs/fine_tuned_results/all_train_acc_%s_s%s_%s_fine_tuned.csv'%(dataset,seed,train_size))
            df_test_auc.to_csv('./outputs/fine_tuned_results/all_test_auc_%s_s%s_%s_fine_tuned.csv'%(dataset,seed,train_size))
            df_test_acc.to_csv('./outputs/fine_tuned_results/all_test_acc_%s_s%s_%s_fine_tuned.csv'%(dataset,seed,train_size))
            df_loss.to_csv('./outputs/fine_tuned_results/all_loss_%s_s%s_%s_fine_tuned.csv'%(dataset,seed,train_size))
            df_raw.to_csv('./outputs/fine_tuned_results/all_raw_%s_s%s_%s_fine_tuned.csv'%(dataset,seed,train_size))


            ## Results of best performing fine tuned checkpoints in training set as an estimation of the model performance in practice
            matrices.append(eval_confusion_matrices[checkpoint])
            eval_aucs.append(aucs[checkpoint])

        df['result'] = [row for row in matrices]
        df['seed'] = random_states
        df = df.set_index('seed')
        df.to_csv('./outputs/raw_%s_fine_tuned_%s.csv'%(dataset,train_size),index=True)

        df_auc['result'] = [row for row in eval_aucs]
        df_auc['seed'] = random_states
        df_auc = df_auc.set_index('seed')
        df_auc.to_csv('./outputs/auc_%s_fine_tuned_%s.csv'%(dataset,train_size),index=True)