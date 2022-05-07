# Importing all the Global libarires required...
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import os
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importing all the Local libaries required...
from preprocessing import data_processing
from model_architecture import NetV1




# Initializing some of the Variables and calling the already written local functions

TRAINING_DIR = ".\..\dataset\BanglaLekha-Isolated\Images"
batch_size = 128
preprocess = data_processing(TRAINING_DIR, batch_size)



# Spilliting the dataset into training and valiidation dataset using the local functions

train_generator = preprocess.train_dataset()
validation_generator = preprocess.train_dataset()



# Calling NetV1() from already written model_architecture python file

architecture = NetV1(input_shape = (32, 32, 1), output = 84)
model = architecture.make_model()



# Initializing the learning rates, loss functions, optimizers and all.

learning_rate = 0.0001
model.compile(loss = 'categorical_crossentropy', 
              metrics = ['accuracy'], 
              optimizer = Adam(learning_rate = learning_rate))
print(model.summary())



# 
trained_model = model.fit(train_generator,
                    epochs = 5,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = validation_generator,
                    validation_steps = validation_generator.samples // batch_size)

# Saving the model for further use
model.save("my_model3.h5")



# Saving the stats files in (.json)
import json

with open("loss_scores","w") as fp:
    json.dump(trained_model.history['loss'], fp)
    fp.close()
    
with open("accuracy_scores","w") as fp:
    json.dump(trained_model.history['accuracy'], fp)
    fp.close()

with open("val_loss_scores","w") as fp:
    json.dump(trained_model.history['val_loss'], fp)
    fp.close()
    
with open("val_accuracy_scores","w") as fp:
    json.dump(trained_model.history['val_accuracy'], fp)
    fp.close()
    
val_true = validation_generator.classes
# val_predict = model.evaluate(validation_generator)
val_predict = model.predict(validation_generator)