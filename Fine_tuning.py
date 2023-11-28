# -*- coding: utf-8 -*-
'''# This program trains a module for reading EEG signals. It is designed to load NP arrays listing the binary files containing the data. 
#It then uses a generator to access the data from the hard drive rather than store it to memmory. once the model has been trained,
 it is tested and a confusion matrix is generated.
'''
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM,GlobalMaxPool2D, MaxPool2D, BatchNormalization
from keras.regularizers import l1

from tensorflow.python import training
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
import numpy as np
import time
import csv
import os

'''# hyperparamaters'''
batch_size=200
number_of_epochs=30
window_size = 600
number_of_channels = 18

"""# Model"""
'''
  longConvLSTM = keras.Sequential([
      layers.Conv2D(filters=32, kernel_size=(3,1) ,input_shape=input_shape),
      layers.Conv2D(filters=16, kernel_size=(5,1)),
      layers.MaxPool2D( pool_size=(42, 2,)),
      layers.Reshape((14,144)),

      layers.LSTM(73, activation='relu' ),
      layers.Dense(73),
      layers.Dense(40),
      layers.Dense(outputs,activation='softmax')
  ])
'''
# Loads a model that already exists. freezes some layers, then trains the model on a tinetuning set. It then prints the results
def fineTune(training_generator,testing_data,testing_labels,name):

  pretrained_model= tf.keras.models.load_model(name+"longConvLSTM")

  # Specify the names or indices of the layers you want to freeze
  layers_to_freeze = [0,1,2,3,4,5] 

  # Freeze the specified layers
  for layer in pretrained_model.layers:
    if layer.name in layers_to_freeze or pretrained_model.layers.index(layer) in layers_to_freeze:
        layer.trainable = False
  
  
  print('\n',pretrained_model.summary())

  pretrained_model.fit(training_generator,
                   epochs = number_of_epochs,
                   verbose = 1)

  
  test_loss, test_acc = pretrained_model.evaluate(testing_data, testing_labels, verbose=2)

  results_string=(str( str(pretrained_model)+':\t test loss:'+str(test_loss)+ ", test acc:"+str(test_acc)+'\n' ))
  print( results_string)
  return results_string,pretrained_model


def fineTuneTesting(testing_data_files):
    dfs=np.empty((1,600,18))
    for file_name in testing_data_files:
        new_data=np.memmap(str(file_name), dtype='float32', mode='r', shape=(600,18))
        new_data=np.array(new_data)
        #here we add the data from each file to one large file
        dfs=np.concatenate((dfs,new_data),axis=0)

    dfs=dfs[1:]
    return dfs
                  


learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.01,
                                                                first_decay_steps=20, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)
opt = keras.optimizers.legacy.Adam(learning_rate=learning_rate)


"""# Generator"""
class My_Custom_Generator(keras.utils.Sequence) :

  def __init__(self, filenames, labels, batch_size) :
    self.filenames = filenames
    self.labels = labels
    self.batch_size = batch_size

  #This defines the number of batches needed to view the entire dataset
  def __len__(self) :
    return int(len(self.filenames) // float(self.batch_size))

  #This returns a tupple with the dataset
  def __getitem__(self, idx) :
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    return np.array([np.memmap(str(file_name), dtype='float32', mode='r', shape=(600,18))
               for file_name in batch_x]), np.array(batch_y)

'''# This prints a report about how the model performed'''
def print_report(strings,name):
  curr_time = str(time.strftime("%m-%d_%H:%M", time.localtime()))
  with open('fine_tune_'+name+'.txt',"w") as writer:

    writer.write('--Accuracy on prooving data--')
    for string in strings:
      print(string)
      writer.write(curr_time)
      writer.write(string)
  writer.close

'''# This prints a confusion matrix about the model'''
def printConfusionMatrix(model, testing_data, testing_labels):
  predicted_labels=model.predict(testing_data)
  predicted_labels=np.argmax(predicted_labels)
  testing_labels=np.argmax(testing_labels)
  confusion_matrix=confusion_matrix(testing_labels,predicted_labels)
  print(confusion_matrix)
  return str(confusion_matrix)

def make_testing_dataset(testing_data):
    testing_dataframe=np.empty((1,600,18))
    for count,file_name in enumerate(testing_data):
        new_dataframe=np.array(np.memmap(str(file_name), dtype='float32', mode='r', shape=(600,18)))
        testing_dataframe=np.concatenate((testing_dataframe,new_dataframe),axis=0)
    return testing_dataframe[1:]

"""# MAIN"""
subjects=('chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10','chb11','chb12','chb13','chb14','chb15','chb16','chb17','chb18','chb19','chb20','chb21','chb22','chb23')
fine_tuning_data=np.array([])
fine_tuning_labels=np.array([0,0])
prooving_data=np.array([])
prooving_labels=np.array([0,0])

for subject in subjects:
    print(subject)
    fine_tuning_data     =np.append( fine_tuning_data,np.load(     (subject+'fine_tuning_data.npy')) )
    fine_tuning_labels   =np.vstack((fine_tuning_labels,np.load(   (subject+'fine_tuning_labels.npy')) ))
    prooving_data   =np.append( prooving_data,np.load(   (subject+'prooving_data.npy')) )
    prooving_labels =np.vstack((prooving_labels,np.load( (subject+'prooving_labels.npy')) ))
    
    #This replaces the indexes to the testing dataset with the testing data itself
    prooving_data=make_testing_dataset(prooving_data)

    os.chdir('Files')
    training_labels=training_labels[1:]
    testing_labels=testing_labels[1:]


    my_training_generator = My_Custom_Generator(fine_tuning_data, fine_tuning_labels, batch_size)
    results_string,pretrained_model= fineTune(my_training_generator,prooving_data,prooving_labels,subject)
    testing_data = fineTuneTesting(testing_data)


    os.chdir('..')
    pretrained_model.save((subject+"fineTunedLSTM"))
    confusion_matrix=printConfusionMatrix(pretrained_model, prooving_data, prooving_labels)
    results_string=results_string+confusion_matrix
    print_report(results_string,subject)