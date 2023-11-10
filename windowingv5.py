# -*- coding: utf-8 -*-
import os
import copy
from pathlib import Path
import inspect


import random as r
import scipy as sp
from scipy.stats import zscore
from scipy.fft import fft, ifft

import sklearn
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss

import numpy as np
from numpy import where
from numpy.random.mtrand import noncentral_chisquare
# Make numpy values easier to read.
np.set_printoptions(precision=6, suppress=True)

"""# Reading Data"""

# This function reads all the bin files for one subject from the hard drive and returns a complete dataset of that subject and that datasets labels.
def readBinFiles(folder):
  dfs=np.empty((1,600,18)) #This array holds all the EEG datasets this program will be using, this creates an zero at the start that is removed
  labels=np.empty((1))
  print(folder)
  for x in os.listdir():
    if x.endswith("labels.npy"):


      new_labels=np.load(x) #here we load the label array from the hard drive, it tells us what bins to read.
      print(x,len(new_labels))

      dfname=x[:-10]+'l' #dfname is the name of the bin file that needs to be read.
      new_labels=new_labels[:-1]
      print(dfname)

      # This reads all the bin filef from the hard drive
      new_data=np.memmap(dfname, dtype='float32', mode='r', shape=(len(new_labels),600,18))
      new_data=np.array(new_data)

      #here we add the data from each file to one large file
      dfs=np.concatenate((dfs,new_data),axis=0)

      labels=np.append(labels,new_labels)

  dfs=dfs[1:]
  print('read\t',dfs.shape,',',len(labels),',',dfs[2][1][3])
  return dfs,labels[1:]



# A function for saving the dataset as binary files.
def saveBins(dfs,labels,folder):
  shape=dfs.shape
  os.chdir('files')
  data_labels=[]
  data_files=[]
  for i in range(0,dfs.shape[0]):
    new_file=np.memmap(folder+"data"+str(i), dtype='float32', mode='w+', shape=(shape[1],shape[2]))
    new_file[:]=dfs[i][:][:]
    new_file.flush()
    #these lines same the arrays so they can be used later.
    data_files=np.append(data_files,folder+"data"+str(i))
    data_labels=np.append(data_labels,labels[i])
    if i%10000==1:
      print(i)
  os.chdir('..')
  return data_labels, data_files

# This function performs a fast faurier transform
def FFT(dfs,labels):
  shape=dfs.shape
  new_df=np.empty((dfs.shape[0],290,dfs.shape[2]))
  for df in range(0, dfs.shape[0]):
    new_df[df]=np.real(fft(dfs[df]))[dfs.shape[1]//2]

  print('FFT\t',new_df.shape,',',len(labels),',',new_df[2][1][3])
  return new_df,labels

def normalize(dfs,labels):
  for df in range(0, dfs.shape[0]):
    dfs[df] = zscore(dfs[df])
  print('Normal\t',dfs.shape,',',len(labels),',',dfs[2][1][3])
  return(dfs,labels)

#This function removes all the seizures from the dataset.
def removeSeizures(dfs,labels,Period_of_interest):
  new_labels=[]
  index_to_delete =[]#This makes a list of all the seizure data that needs to be delieted.

  for i in range(0,len(labels)):
    #if the number is less than 0, it is in a seizure and is removed
    if (labels[i] <0):
      index_to_delete.append(i)
      new_labels.append(2) #The twos should all be removed at the end of this function.
    #if the label is 0, it is after the last siezure, i don;t really know how to treat it
    elif (labels[i]==0):
      new_labels.append(0)

    #if the label is greater than the poi, then it is far from the seizure and is marked with a 0
    elif(labels[i] >Period_of_interest):
      new_labels.append(0)

    #If none of the others were true, it must be in the period of interest, and it gets marked with a 1
    else:
      new_labels.append(1)

  #this deletes all the seizures found in the previous loop.
  new_labels=np.array(new_labels)
  dfs=np.delete(dfs,index_to_delete,axis=0)
  new_labels=np.delete(new_labels,index_to_delete)
  print('remove\t',dfs.shape,',',len(new_labels),',',dfs[2][1][3])
  return dfs,new_labels

# a function for balancing the labels in the dataset
def balance(dfs,labels):
  shape=dfs.shape

  # summarize class distribution
  counter = Counter(labels)
  print("before:\t",counter,(counter[1])/len(dfs) )
  under=RandomUnderSampler(sampling_strategy=.45)# An undersampling strategy
  nr = NearMiss(sampling_strategy=.60)           # An undersampling strategy
  over=SMOTE(sampling_strategy=.96)              # An oversampling strategy


  #These two if statements prevent the rebalancing from being called if the data is already balanced
  if ((counter[1])/len(dfs))<=.40:
    dfs=dfs.reshape(-1,shape[1]*shape[2]) #iblearn wants a specific shape of data,
    dfs, labels = under.fit_resample(dfs, labels )
  else:print('skipped-U')

  counter = Counter(labels)
  if ((counter[1])/len(dfs))<=.49:
    dfs=dfs.reshape(-1,shape[1]*shape[2])
    dfs, labels=over.fit_resample(dfs, labels)

    counter = Counter(labels)
    print("after:\t",counter,(counter[1])/len(dfs) )
  else: print('skipped-O')

  dfs=dfs.reshape(-1,shape[1],shape[2]) #put the shape back
  print('balance\t',dfs.shape,',',len(labels),',',dfs[2][1][3])
  return dfs,labels

# A function for shuffleing the dataset
def shuffle(dfs,labels):
  dfs,labels=sklearn.utils.shuffle(dfs,labels)
  print('shuffle\t',dfs.shape,',',len(labels),',',dfs[2][1][3])
  return dfs,labels

#This function converts binary integer arrays to one-hot arrays
def oneHot(labels):
  zero_vector=[1,0]
  one_vector =[0,1]
  new_array=[]
  for i in labels:
    if i==0:
        new_array.append(zero_vector)
    else :
        new_array.append(one_vector)
  new_array=np.array(new_array)
  return new_array

#This function saves an np array
def saveArray(qualifier,data_files):
  for files in data_files:
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    file_name= str([files_name for files_name, files_val in callers_local_vars if files_val is files][0])
    print(qualifier+file_name)
    np.save( qualifier+file_name , files)

def trainingDataPipeline(dfs,labels):
  #dfs,labels=FFT(dfs,labels)
  dfs,labels=balance(dfs,labels)
  dfs,labels=normalize(dfs,labels)
  labels=oneHot(labels)
  return dfs, labels

def testingDataPipeline():
  dfs,labels=normalize(dfs,labels)
  labels=oneHot(labels)
  testing_data, validation_data, testing_labels, validation_labels = train_test_split(dfs,labels, test_size=0.5, random_state=42)
  return testing_data, validation_data, testing_labels, validation_labels

def fineTuningPipeline(dfs,labels):
  dfs,labels=normalize(dfs,labels)
  one_hot_labels=oneHot(labels)
  fine_tuning_data, prooving_data, fine_tuning_labels, prooving_labels = train_test_split(dfs,labels, test_size=0.01, random_state=42, shuffle=False)


"""# MAIN"""
#This is the period before a seizure that will be examined.
period_of_interest=60*60*1

folders_for_making_model = ('chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10','chb11','chb12','chb13','chb14','chb15','chb16','chb17','chb18','chb19','chb20','chb21','chb22','chb23')

#This for loop processes the data for the all subjects who are not being excluded
for index,folder in enumerate(folders_for_making_model):

  os.chdir(folder)
  #this is the code for creating the large dataset from many subjects
  dfs,labels=readBinFiles(folder)
  dfs,labels=removeSeizures(dfs,labels, period_of_interest)
  dfs,labels=shuffle(dfs,labels)
  training_data, X_test, training_labels, y_test = train_test_split(dfs,labels, test_size=0.1, random_state=42)
  training_data,training_labels=trainingDataPipeline(training_data,training_labels)

  testing_data, validation_data, testing_labels, validation_labels=testingDataPipeline(X_test,y_test)

  os.chdir('..')
  data_labels, data_files = saveBins(dfs,labels,folder)
  dfs=0
  saveArray(folder,(testing_data, validation_data, testing_labels, validation_labels,training_data,training_labels))
