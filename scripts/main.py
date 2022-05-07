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

from model_architecture import AKHCRNetV1
from preprocessing import data_processing


TRAINING_DIR = ".\dataset\BanglaLekha-Isolated\Images"
preprocess = data_processing(TRAINING_DIR)
train_generator, validation_generator = preprocess.train_validate()


architecture = AKHCRNetV1(input_shape = (32, 32, 1), output = 84)
model = architecture.make_model()

learning_rate = 0.0004
batch_size = 512
model.compile(loss = 'categorical_crossentropy',
                metrics = ['accuracy'],
                optimizer = Adam(learning_rate = learning_rate))

model.summary()
print("Number of hidden layers : ", len(model.layers))




history = model.fit(train_generator,
                    epochs = 5,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = validation_generator,
                    validation_steps = validation_generator.samples // batch_size)
model.save('my_model2.h5')

