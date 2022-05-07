
""" 
This respective modules is meant to design a 32 layers deep neural network based on tensorflow to correcly predict the bengali handwritten-characters. 
"""


# Importing all the required libaries...
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from keras.models import Model


# Creating the class for designing a DNN
class NetV1:
    """ 
    The particular class takes two arguments:
                    input_shape    : it takes the size the imput images
                    output         : it takes the number of outputs (that they want the model to return after training)
    The particular class or module contains two function/method:
                    make_model     : which returns the build architecture of the network
    """

    def __init__(self, input_shape, output):
        """ Intialzing Method """
        
        # Intializng the input variables
        self.input_shape = input_shape
        self.output = output

    def make_model(self):
        """ Bulding the architecture of the neural network, and returing the buld model. """
        
        # Defining the input size/shape of the images
        input_shape = self.input_shape
        input_layer = Input(input_shape)
        
        # Configuring all the other required layers
        conv2D_1 = Conv2D(32, (5, 5), activation = 'relu', padding = 'same', name = 'conv2D_1')(input_layer)
        conv2D_2 = Conv2D(32, (5, 5), padding = 'same', name = 'conv2D_2')(conv2D_1)
        batchnorm_1 = BatchNormalization(name = 'first_batchNorm_layer')(conv2D_2)
        activation1 = Activation('relu')(batchnorm_1)

        maxpool_1 = MaxPooling2D(pool_size = (2, 2))(batchnorm_1)

        first_branch_0_conv2D_1 = Conv2D(128, (1, 1), activation = 'relu', padding = 'same',
                                         name = 'first_branch_0_conv2D_1')(maxpool_1)
        first_branch_0_conv2D_2 = Conv2D(128, (5, 5), activation = 'relu', padding = 'same',
                                         name = 'first_branch_0_conv2D_2')(first_branch_0_conv2D_1)

        first_branch_1_conv2D_1 = Conv2D(128, (1, 1), activation = 'relu', padding = 'same',
                                         name = 'first_branch_1_conv2D_1')(maxpool_1)
        first_branch_1_conv2D_2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same',
                                         name = 'first_branch_1_conv2D_2')(first_branch_1_conv2D_1)

        first_branch_2_conv2D = Conv2D(128, (1, 1), activation = 'relu', padding = 'same',
                                       name = 'first_branch_2_conv2D')(maxpool_1)

        first_branch_3_MaxPool_1 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same',
                                                name = 'first_branch_3_MaxPool_1')(maxpool_1)
        first_branch_3_Convolution = Conv2D(64, (1, 1), padding = 'same', activation = 'relu',
                                            name = 'first_branch_3_Convolution')(first_branch_3_MaxPool_1)

        concatened_first_branch = Concatenate()(
            [first_branch_0_conv2D_2, first_branch_1_conv2D_2, first_branch_2_conv2D, first_branch_3_Convolution])
        concatenation_activation = Activation('relu')(concatened_first_branch)

        conv2D_3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'conv2D_3')(concatenation_activation)
        maxpool_2 = MaxPooling2D(pool_size = (2, 2))(conv2D_3)
        conv2D_4 = Conv2D(256, (3, 3), padding = 'same', name = 'conv2D_4')(maxpool_2)
        batchnorm_2 = BatchNormalization(name = 'second_batchNorm_layer')(conv2D_4)
        activation2 = Activation('relu')(batchnorm_2)

        conv2D_5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'conv2D_5')(activation2)
        maxpool_3 = MaxPooling2D(pool_size = (2, 2))(conv2D_5)
        conv2D_6 = Conv2D(512, (3, 3), padding = 'same', name = 'conv2D_6')(maxpool_3)
        batchnorm_3 = BatchNormalization(name = 'third_batchNorm_layer')(conv2D_6)
        activation3 = Activation('relu')(batchnorm_3)
        maxpool_4 = MaxPooling2D(pool_size = (2, 2))(activation3)

        flattened_before_dense = Flatten()(maxpool_4)
        dense1 = Dense(1024, activation='relu', name = 'firstDenseLayer',
                       kernel_regularizer = keras.regularizers.l2(0.001))(flattened_before_dense)
        dense2 = Dense(512, activation = 'relu', name = 'SecondDenseLayer',
                       kernel_regularizer = keras.regularizers.l2(0.001))(dense1)
        dropout1 = Dropout(0.5, name = 'FirstDropOutLayer')(dense2)
        dense3 = Dense(256, activation = 'relu', name = 'ThirdDenseLayer',
                       kernel_regularizer = keras.regularizers.l2(0.001))(dropout1)

        dense4 = Dense(128, activation = 'relu', name = 'FourthDenseLayer')(dense3)

        prediction_branch = Dense(self.output, activation = 'softmax', name = 'FinalSoftmaxLayer')(dense4)

        model = Model(inputs = input_layer, outputs = prediction_branch)

        return model

