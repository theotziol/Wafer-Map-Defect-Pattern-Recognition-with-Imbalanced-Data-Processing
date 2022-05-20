import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from copy import deepcopy as dc
import random
import time
import preprocessing
import models
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, RepeatedKFold
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, Add, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, Convolution2DTranspose
#import the dataset
df=pd.read_pickle("LSWMD.pkl" )
df.drop(['waferIndex','lotName','trianTestLabel'], axis = 1, inplace = True)


def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df['waferMap'].apply(find_dim)


df['failureNum']=df['failureType']
mapping_type={'Center':0,
              'Donut':1,
              'Edge-Loc':2,
              'Edge-Ring':3,
              'Loc':4,
              'Random':5,
              'Scratch':6,
              'Near-full':7,
              'none':8}
df=df.replace({'failureNum':mapping_type})

df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)].reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)].reset_index()
df_nonpattern = df[(df['failureNum']==8)].reset_index()
del df


df_center = df_withpattern[(df_withpattern['failureNum']==0)].reset_index()
df_Donut = df_withpattern[(df_withpattern['failureNum']==1)].reset_index()
df_Edge_Loc = df_withpattern[(df_withpattern['failureNum']==2)].reset_index()
df_Edge_Ring = df_withpattern[(df_withpattern['failureNum']==3)].reset_index()
df_Loc = df_withpattern[(df_withpattern['failureNum']==4)].reset_index()
df_Random = df_withpattern[(df_withpattern['failureNum']==5)].reset_index()
df_Scratch = df_withpattern[(df_withpattern['failureNum']==6)].reset_index()
df_Near_full = df_withpattern[(df_withpattern['failureNum']==7)].reset_index()



#Preprocessing
none_downsampled = preprocessing.downsample(df_nonpattern, 15000)
#change dims
df_Donut['waferMap'] = df_Donut['waferMap'].apply(preprocessing.resize)
df_Edge_Loc['waferMap'] = df_Edge_Loc['waferMap'].apply(preprocessing.resize)
df_Edge_Ring['waferMap'] = df_Edge_Ring['waferMap'].apply(preprocessing.resize)
df_Loc['waferMap'] = df_Loc['waferMap'].apply(preprocessing.resize)
df_Near_full['waferMap'] = df_Near_full['waferMap'].apply(preprocessing.resize)
df_Random['waferMap'] = df_Random['waferMap'].apply(preprocessing.resize)
df_Scratch['waferMap'] = df_Scratch['waferMap'].apply(preprocessing.resize)
df_center['waferMap'] = df_center['waferMap'].apply(preprocessing.resize)
none_downsampled['waferMap'] = none_downsampled['waferMap'].apply(preprocessing.resize)

#apply different splits
none_downsampled_tr, none_downsampled_ts = preprocessing.split(none_downsampled, 0.9, False)
df_Donut_tr, df_Donut_ts = preprocessing.split(df_Donut, 0.7)
df_Edge_Loc_tr, df_Edge_Loc_ts = preprocessing.split(df_Edge_Loc, 0.9)
df_Edge_Ring_tr, df_Edge_Ring_ts = preprocessing.split(df_Edge_Ring, 0.9)
df_Loc_tr, df_Loc_ts = preprocessing.split(df_Loc, 0.9)
df_Near_full_tr, df_Near_full_ts = preprocessing.split(df_Near_full, 0.7)
df_Random_tr, df_Random_ts = preprocessing.split(df_Random,0.8)
df_Scratch_tr, df_Scratch_ts = preprocessing.split(df_Scratch,0.8)
df_center_tr, df_center_ts = preprocessing.split(df_center, 0.9)

#apply different augmentations to all except none-pattern and edge ring

df_Donut_tr = preprocessing.augment(df_Donut_tr,3)
df_Edge_Loc_tr = preprocessing.augment(df_Edge_Loc_tr,2)
df_Loc_tr = preprocessing.augment(df_Loc_tr,2)
df_Near_full_tr = preprocessing.augment(df_Near_full_tr,3)
df_Random_tr = preprocessing.augment(df_Random_tr,3)
df_Scratch_tr = preprocessing.augment(df_Scratch_tr,3)
df_center_tr = preprocessing.augment(df_center_tr,1)


training = pd.concat([none_downsampled_tr,
                            df_Donut_tr,
                            df_Edge_Loc_tr,
                            df_Edge_Ring_tr,
                            df_Loc_tr,
                            df_Near_full_tr,
                            df_Random_tr,
                            df_Scratch_tr,
                            df_center_tr], ignore_index = True)
training = training.sample(frac= 1)


testing = pd.concat([none_downsampled_ts,
                            df_Donut_ts,
                            df_Edge_Loc_ts,
                            df_Edge_Ring_ts,
                            df_Loc_ts,
                            df_Near_full_ts,
                            df_Random_ts,
                            df_Scratch_ts,
                            df_center_ts], ignore_index = True)
testing = testing.sample(frac= 1)

#training
x = np.asarray(training['waferMap'].values)
y = to_categorical(training['failureNum'].values, num_classes = 9)
#reshape x from (batch_size, ) to (batch_size, width, height, channels)
lis = []
for i in range(len(x)):
    lis.append(x[i])

train = np.asarray(lis)
del x
#testing
x_t = np.asarray(testing['waferMap'].values)
y_test = to_categorical(testing['failureNum'].values, num_classes = 9)
#reshape x from (batch_size, ) to (batch_size, width, height, channels)
test =[]
for i in range(len(x_t)):
    test.append(x_t[i])
x_test = np.asarray(test)
del x_t


def evaluate_model(trx, trainy,test_x, test_y, n_folds = 5):
    scores_a,scores_p,scores_r = list(), list(), list()

    histories = []
    m = []
    kfold = KFold(n_folds,shuffle=True)
    for train_ix, test_ix in kfold.split(trx):
    #for i in range(n_folds):
        model = models.new_model()
        train_x, train_y = trx[train_ix],trainy[train_ix]
        eval_x, eval_y = trx[test_ix],trainy[test_ix]
        batch = 256
        eval_batch = 128
        callbacks = [keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True), keras.callbacks.LearningRateScheduler(scheduler)]
        #history = model.fit(train, y, epochs=1200, shuffle= True,batch_size = batch, validation_batch_size = eval_batch, callbacks=callbacks, validation_split = 0.2, verbose = 0)
        history = model.fit(train_x,train_y,epochs=1500,batch_size = batch, validation_batch_size = eval_batch, callbacks=callbacks, validation_data= (eval_x,eval_y), verbose = 0)
        l, a, p, r = model.evaluate(test_x,test_y) 
        scores_a.append(a)
        scores_p.append(p)
        scores_r.append(r)
        histories.append(history)
        m.append(model)
    return scores_a,scores_p,scores_r,histories, m


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return 0.00001

