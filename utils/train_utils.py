
import tensorflow as tf
import numpy as np
import pickle

from utils.loss import CrossEntropy2d
from dataset.ade20_dataset import VOCDataSet, VOCGTDataSet, VOCDataTestSet


def loss_function(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = tf.cast(label, tf.int32)
    return CrossEntropy2d()(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(args, label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return tf.convert_to_tensor(one_hot, dtype=tf.float32)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = tf.convert_to_tensor(D_label, dtype=tf.float32)
    return D_label


def restore_model_from_checkpoint(checkpoint_path, model):
    """
    Restores the pretrained parameters from the checkpoint to the model.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    restore_dict = reader.get_variable_to_shape_map()

    for var in model.variables:
       if var.path in restore_dict:
            tensor = reader.get_tensor(var.path)
            var.assign(tensor)
            print(f'Successfully loaded {var.path} into the model.')
            
    print("Model restored with subset matching from:", checkpoint_path)


def compile_model(model_S, model_D, args):
    """
    Compiles the segmentation and discriminator models.
    """
    restore_model_from_checkpoint(args.restore_from, model_S)
    optim_S = tf.keras.optimizers.SGD(learning_rate=args.LEARNING_RATE, momentum=args.momentum, decay=args.weight_decay)
    optim_D = tf.keras.optimizers.Adam(learning_rate=args.learning_rate_D, beta_1=0.9, beta_2=0.99)
    model_S.compile(
        optimizer   = optim_S,
        loss        = loss_function,
    ) 

    model_D.compile(
        optimizer   = optim_D,
        loss        = loss_function,
    ) 

def load_ade20(args):
    """
    Loads ade20 dataset.
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    IMG_MEAN = np.array((123.68748388, 118.66391674, 109.94100899), dtype=np.float32)

    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)
    print(train_dataset_size)

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    
    if args.partial_data is None:
        trainloader = tf.data.Dataset.from_tensor_slices(train_dataset)
        trainloader = trainloader.shuffle(buffer_size=len(train_dataset))
        trainloader = trainloader.batch(batch_size=args.batch_size)
    
    else: #sample partial data
        partial_size = int(args.partial_data * train_dataset_size)
        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id))
            print('loading train ids from {}'.format(args.partial_id))
        else:
            train_ids = range(train_dataset_size)
            np.random.shuffle(train_ids)

        # Calculate the size of partial data
        partial_size = int(args.partial_data * train_dataset_size)

        # Split train_ids into partial and remaining parts
        train_ids_partial = train_ids[:partial_size]
        train_ids_remain = train_ids[partial_size:]

        # Create TensorFlow datasets for partial and remaining data
        train_dataset_partial = tf.data.Dataset.from_tensor_slices(train_ids_partial)
        train_dataset_remain = tf.data.Dataset.from_tensor_slices(train_ids_remain)
        # Create dataset objects for partial, remaining, and ground truth data loaders
        trainloader = (train_dataset_partial
                            .shuffle(buffer_size=partial_size)
                            .batch(batch_size=args.batch_size)
                            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                            )

        trainloader_remain = (train_dataset_remain
                            .shuffle(buffer_size=len(train_ids_remain))
                            .batch(batch_size=args.batch_size)
                            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                            )

        trainloader_gt = (train_gt_dataset
                        .shuffle(buffer_size=len(train_ids_partial))
                        .batch(batch_size=args.batch_size)
                        # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                        )
    return trainloader, trainloader_gt, trainloader_remain
    