
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
    res2a_branch1 = Conv2D(64, 3, 3, activation='relu', name='res2a_branch2a')
    res2a_branch1.add_weight(name = 'weights', shape =(1, 1, 64, 64),initializer='glorot_uniform',trainable=True)
    x = res2a_branch1(x)
    x = BatchNormalization(name = 'bn2b_branch2a')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    res2a_branch2b = Conv2D(64, 3, 3, activation='relu', name='res2a_branch2b')
    res2a_branch2b.add_weight(name = 'weights', shape =(3, 3, 64, 64),initializer='glorot_uniform',trainable=True)
    x = res2a_branch2b(x)

    res2a_branch2c = Conv2D(64, 3, 9, name='res2a_branch2c', padding='same')
    res2a_branch2c.add_weight(name = 'weights', shape =[1, 1, 64, 256],initializer='glorot_uniform',trainable=True)
    res = res2a_branch2c(res)
    res = BatchNormalization(name = 'bn2b_branch2b')(res)
    x = ReLU()(x + res)

    # Res Block 2
    res = x
    x = ZeroPadding2D(padding=(1, 1))(x)
    res2b_branch2a = Conv2D(128, 3, 3, activation='relu', name='res2b_branch2a')
    res2b_branch2a.add_weight(name = 'weights', shape =[1, 1, 256, 64],initializer='glorot_uniform',trainable=True)
    x = res2b_branch2a(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    res2b_branch2b = Conv2D(128, 3, 3, activation='relu', name='res2b_branch2b')
    res2b_branch2b.add_weight(name = 'weights', shape =[3, 3, 64, 64],initializer='glorot_uniform',trainable=True)
    x = res2b_branch2b(x)

    x = BatchNormalization(name = 'bn3a_branch2a')(x)
    res2b_branch2c = Conv2D(64, 3, 9, padding='same', name='res2b_branch2c')
    res2b_branch2c.add_weight(name = 'weights', shape =[1, 1, 64, 256],initializer='glorot_uniform',trainable=True)
    x = res2b_branch2c(x)

    res = BatchNormalization(name = 'bn2c_branch2b')(res)
    p2 = ReLU()(x + res)
  
    # Prediction Block
    b1 = ZeroPadding2D(padding=(6, 6))(p2)
    res5a_branch2a = Conv2D(256, kernel_size=3, dilation_rate=(6, 6), activation='relu', name='res5a_branch2a')
    res5a_branch2a.add_weight(name = 'weights', shape =[1, 1, 1024, 512],initializer='glorot_uniform',trainable=True)
    b1 = res5a_branch2a(b1)

    b1 = Dropout(0.5)(b1)
    res5a_branch2b = Conv2D(256, 1, 1, activation='relu', name='res5a_branch2b')
    res5a_branch2b.add_weight(name = 'weights', shape =[3, 3, 512, 512],initializer='glorot_uniform',trainable=True)
    b1 = res5a_branch2b(b1)

    b1 = Dropout(0.5)(b1)
    res5a_branch2c = Conv2D(num_classes, 1, 1, activation='relu', name='res5a_branch2c')
    res5a_branch2c.add_weight(name = 'weights', shape =[1, 1, 512, 2048],initializer='glorot_uniform',trainable=True)
    b1 = res5a_branch2c(b1)

    resized = UpSampling2D(size=(32, 32))(b1)
    logits = Softmax()(resized)
    model = tf.keras.Model(inputs, logits, name='myDeeplab')
    return model

# deeplab = myDeeplab(input_shape=(256, 256, 3))
# deeplab.summary()