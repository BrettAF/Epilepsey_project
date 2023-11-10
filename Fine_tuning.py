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
def funLongConvLSTM(my_training_generator,window_size,number_of_channels,my_validation_generator,testing_data):
  number_of_channels=18

  outputs=2
  input_shape = ( window_size, number_of_channels, 1)

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
  longConvLSTM.compile(
                  optimizer='Adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

  print('\n',longConvLSTM.summary())

  longConvLSTM.fit(my_training_generator,
                   epochs = number_of_epochs,
                   verbose = 1, validation_data=my_validation_generator)

  
  test_loss, test_acc = longConvLSTM.evaluate(testing_data, verbose=2)

  results_string=(str( str(longConvLSTM)+':\t test loss:'+str(test_loss)+ ", test acc:"+str(test_acc)+'\n' ))
  print( results_string)
  return results_string,longConvLSTM

def fineTuneTesting(testing_data_files):
    dfs=np.empty((1,600,18))
    for file_name in testing_data_files:
        new_data=np.memmap(str(file_name), dtype='float32', mode='r', shape=(600,18))
        new_data=np.array(new_data)

        #here we add the data from each file to one large file
        dfs=np.concatenate((dfs,new_data),axis=0)

    dfs=dfs[1:]
    return dfs
                  

"""# Generator"""
learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.01,
                                                                first_decay_steps=20, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)
opt = keras.optimizers.legacy.Adam(learning_rate=learning_rate)

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
  with open('general_'+name+'.txt',"w") as writer:

    writer.write('--Accuracy on testing data--')
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
for one_out in subjects:

    

    training_data   =np.load( (one_out+'FineTraining_data_files.npy')) 
    training_labels =np.load( (one_out+'Finetraining_file_labels.npy')) 
    testing_data    =np.load( (one_out+'Finetesting_data_files.npy')) 
    testing_labels  =np.load( (one_out+'Finetesting_file_labels.npy')) 
    vailidation_data=np.load( (one_out+'Finevalidation_data_files.npy')) 
    vailidation_labels=np.load( (one_out+'Finevalidation_file_labels.npy')) 
    testing_times= np.load( (one_out+'testing_times.npy')) 
    os.chdir('Files')


    my_training_generator = My_Custom_Generator(training_data, training_labels, batch_size)
    my_validation_generator = My_Custom_Generator(vailidation_data,vailidation_labels,batch_size)
    testing_data = fineTuneTesting(testing_data)


    results_string,model=funLongConvLSTM(my_training_generator,window_size,number_of_channels,my_validation_generator)
    os.chdir('..')
    model.save((one_out+"longConvLSTM"))
    confusion_matrix=printConfusionMatrix(model, testing_data, testing_labels)
    results_string=results_string+confusion_matrix
    print_report(results_string,one_out)
#make_testing_dataset(testing_data)
#printConfusionMatrix(model,testing_data,testing_labels)