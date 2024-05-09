import tensorflow as tf

class FCDiscriminator(tf.keras.Model):
    '''
    This class represents a fully-convoluational network discriminator.
    '''
    def __init__(self, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(ndf, kernel_size=4, strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(ndf*2, kernel_size=4, strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(ndf*4, kernel_size=4, strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(ndf*8, kernel_size=4, strides=2, padding='same')
        self.classifier = tf.keras.layers.Conv2D(1, kernel_size=4, strides=2, padding='same')

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.logits_layer = tf.keras.activations.sigmoid
        self.upsampling = tf.keras.layers.UpSampling2D(size=(32, 32))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.logits_layer(x)
        x = self.upsampling(x)

        return x