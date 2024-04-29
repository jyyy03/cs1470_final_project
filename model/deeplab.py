import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(inputs, filters, strides=1):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same', use_bias=False)(inputs)

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, training=False):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out
    



class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=1, padding='same', dilation_rate=dilation, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=planes * 4, kernel_size=1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, training=False):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out
    
class Classifier_Module(tf.keras.Model):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = []
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(tf.keras.layers.Conv2D(num_classes, kernel_size=3, strides=1, padding=padding, dilation_rate=dilation, use_bias=True))

        for conv in self.conv2d_list:
            conv.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

    def call(self, x, training=False):
        out = self.conv2d_list[0](x)
        for i in range(1, len(self.conv2d_list)):
            out += self.conv2d_list[i](x)
        return out

class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # tf.keras.layers.Conv2D(3, 64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],["same"], num_classes)

        # TODO Maybe need to add something like:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization()])
        layers = [block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return tf.keras.Sequential(layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer5(x, training=training)

        return x
    
    def get_1x_lr_params_NOscale(model):
      """
      This function returns all the parameters of the net except for 
      the last classification layer. Note that for each batch normalization layer, 
      requires_grad is set to False, therefore this function does not return 
      any batch normalization parameter.
      """
      params = []
      params.append(model.conv1.trainable_variables)
      params.append(model.bn1.trainable_variables)
      for layer in model.layers:
          if isinstance(layer, BasicBlock) or isinstance(layer, Bottleneck):
              params.extend(layer.trainable_variables)
      return params

    def get_10x_lr_params(model):
        """
        This function returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes.
        """
        return model.layer5.trainable_variables
    
    
    def optim_parameters(self, model, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(model), 'lr': learning_rate},
            {'params': self.get_10x_lr_params(model), 'lr': 10 * learning_rate}]


def Res_Deeplab(num_classes=11):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
