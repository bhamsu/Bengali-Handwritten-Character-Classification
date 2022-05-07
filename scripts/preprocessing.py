
""" 
This respective modules is designed to pre-process all the image dataset with a particular batch size, and returning training & validation dataset.This module uses ImageDataGenerator of tensorflow libary for rescaling, sampling etc. purposes. 
"""


# Importing all the requried libarirs...
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


# Creating the class for imgae data preprocessing
class data_processing:
    """ 
    The particular class takes only one argument:
                    TRAINING_DIR    : it takes the path as an input
    The particular class or module contains two function/method:
                    train_dataset   : which returns the training datset
                    validate_dataset: which returns the validation datset
    """

    def __init__(self, TRAINING_DIR, batch_size = 256):
        """ Intialzing Method """
        
        # Intializng the input variables
        self.TRAINING_DIR = TRAINING_DIR
        self.batch_size = batch_size
        self.train_datagen = ImageDataGenerator(rescale = 1. / 255,
                                           # shear_range = 0.2,
                                           # zoom_range = 0.2,
                                           horizontal_flip = False,
                                           validation_split = 0.28)  # set validation split

    def train_dataset(self):
        """ Returning the training dataset. """
        
        # Using the tensorflow.keras.preprocessing.image.ImageDataGenerator object
        train_generator = self.train_datagen.flow_from_directory(
            directory = self.TRAINING_DIR,
            target_size = (32, 32),
            color_mode = 'grayscale',
            class_mode = 'categorical',
            batch_size = self.batch_size,
            subset = 'training')  # set as training data
        
        return train_generator
    
    def validation_dataset(self):
        """ Returning the validation dataset. """
        
        # Using the tensorflow.keras.preprocessing.image.ImageDataGenerator object
        validation_generator = self.train_datagen.flow_from_directory(
            directory = self.TRAINING_DIR,  # same directory as training data
            target_size = (32, 32),
            color_mode = 'grayscale',
            class_mode = 'categorical',
            batch_size = self.batch_size,
            subset = 'validation')  # set as validation data

        return validation_generator
    
    def __del__(self):
        """ Calling the Destructor. """
        pass