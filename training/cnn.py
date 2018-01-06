import keras
from keras import layers
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.regularizers import l2


def CNN_trainer(input_shape, num_classes):
    model = Sequential()  # model is a linear stack of layers.

    '''
    Explanations for convolutional layer:
            Generates feature maps that represent how pixel values are enhanced, for example, edge and pattern detection.
            filters:Number of convolution filters to use.
            kernel_size:    Number of convolution kernel.
            padding:Same means no padding. no padding because output does not depend input since we're using same input shape for all.
            name:   Name of initialization. Only relevant if no argument is passes whic is not our case
            input_shape:1X48X48, from ferdatset imagesize
    Normalization layer:
            BatchNormalization: Applies a transformation that maintains the mean activation close to 0 
                                and the activation standard deviation close to 1.
    Activation layer:
            Activation: non-linear regression
    Pooling layer:
            pool_size:  integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). 
            (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same 
            window length will be used for both dimensions.
            It is a dimension reduction technique usually applied after one or several convolutional layers. 
            It is an important step when building CNNs as adding more convolutional layers can greatly affect computational time
    Core layer: 
            DropOut:    Dropout consists in randomly setting a fraction rate of input units to 0 
            at each update during training time, which helps prevent overfitting.

    '''

    # First box
    #convolutional layer & normalization layer
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same', name='image_array', input_shape=input_shape))
    model.add(BatchNormalization()) # batch normalization is seems like a layer but just transforms the input array.
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    model.add(BatchNormalization()) # batch normalization is seems like a layer but just transforms the input array.
    # activation layer
    model.add(Activation('relu'))   # non-linear 
    # pooling layer
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same')) # dimension reduction, keep max value pixel in 2x2 dimension
    #dropout layer
    model.add(Dropout(.5))

    # Second box
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    # Third box
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    # Fourth box
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    #Final box
    #No avarage, no pooling
    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax',name='predictions'))
    return model

if __name__ == "__main__":
    input_shape = (64, 64, 1) #input image shape
    model = CNN_trainer((48, 48, 1), 7) #7 emotions, 48x48 image size from model, grayscale not rgb
    model.summary()
