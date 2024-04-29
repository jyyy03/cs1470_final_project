
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, ReLU, Dropout, Softmax, UpSampling2D, Add


def myDeeplab(input_shape, num_classes= 2):
    """
    Txis model class will contain txe arcxitecture for your CNN txat 
    classifies images. We xave left in variables in txe constructor
    for you to fill out, but you are welcome to cxange txem if you'd like.
    """
    inputs =  Input(shape=input_shape)

    # Init Block
    x = ZeroPadding2D(padding=3)(inputs)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias= False)(x)
    x = BatchNormalization(name = 'bn1')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Res Block 1
    res = x
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, 3, 3, activation='relu', name='conv1_1')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, 3, 3, name='conv1_2')(x)
    x = BatchNormalization()(x)
    res = Conv2D(64, 3, 9, padding='same')(res)
    res = BatchNormalization()(res)
    x = ReLU()(x + res)

    # Res Block 2
    res = x
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, 3, 3, activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, 3, 3, name='conv2_2')(x)
    x = BatchNormalization()(x)
    res = Conv2D(128, 3, 9, padding='same')(res)
    res = BatchNormalization()(res)
    p2 = ReLU()(x + res)
  
    # Prediction Block
    b1 = ZeroPadding2D(padding=(6, 6))(p2)
    b1 = Conv2D(256, kernel_size=3, dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(256, 1, 1, activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(num_classes, 1, 1, activation='relu', name='fc8_voc12_1')(b1)

    resized = UpSampling2D(size=(input_shape[0], input_shape[1]))(b1)
    logits = Softmax()(resized)
    model = tf.keras.Model(inputs, logits, name='myDeeplab')
    return model

# deeplab = myDeeplab(input_shape=(256, 256, 3))
# deeplab.summary()