import tensorflow as tf

class CrossEntropy2d(tf.keras.layers.Layer):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def call(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert predict.shape.rank == 4
        assert target.shape.rank == 3
        assert predict.shape[0] == target.shape[0], "{0} vs {1}".format(predict.shape[0], target.shape[0])
        assert predict.shape[2] == target.shape[1], "{0} vs {1}".format(predict.shape[2], target.shape[1])
        assert predict.shape[3] == target.shape[2], "{0} vs {1}".format(predict.shape[3], target.shape[2])
        
        n, c, h, w = predict.shape
        target_mask = tf.math.logical_and(target >= 0, target != self.ignore_label)

        target = tf.boolean_mask(target, target_mask)
        if target.shape.ndims == 0:
            return tf.zeros([1])
        predict = tf.transpose(predict, perm=[0, 2, 3, 1])      
        target_mask_reshaped = tf.reshape(target_mask, [n, h, w, 1])
        target_mask_repeated = tf.tile(target_mask_reshaped, [1, 1, 1, c])
        predict = tf.boolean_mask(predict, target_mask_repeated)
        predict = tf.reshape(predict, (-1, c))

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=predict))
        if weight is not None:
            loss = tf.reduce_mean(loss * weight)   
                 
        if self.size_average:
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.reduce_sum(loss)
        return loss


class BCEWithLogitsLoss2d(tf.keras.layers.Layer):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def call(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert predict.shape.rank == 4
        assert target.shape.rank == 4
        assert predict.shape[0] == target.shape[0], "{0} vs {1}".format(predict.shape[0], target.shape[0])
        assert predict.shape[2] == target.shape[2], "{0} vs {1}".format(predict.shape[2], target.shape[2])
        assert predict.shape[3] == target.shape[3], "{0} vs {1}".format(predict.shape[3], target.shape[3])
        n, c, h, w = predict.shape
        target_mask = tf.math.logical_and(target >= 0, target != self.ignore_label)
        target = tf.boolean_mask(target, target_mask)
        if target.shape.ndims == 0:
            return tf.zeros([1])
        predict = tf.boolean_mask(predict, target_mask)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=predict)
        if weight is not None:
            loss = tf.multiply(loss, weight)

        if self.size_average:
            return tf.reduce_mean(loss)
        else:
            return tf.reduce_sum(loss)