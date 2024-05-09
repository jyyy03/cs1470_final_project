import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_DIR = 'dataset/Forest/Forest Segmented/Forest Segmented'

def preprocess_helper(image_path, mask_path, target_size=(321, 321)):
    '''
    Helper function for preprocessing the dataset.
    '''
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0  # Normalize mask
    
    # Generate two masks
    mask_forest = tf.where(mask > 0.5, 1.0, 0.0)  # Highlight forest
    mask_not_forest = tf.where(mask <= 0.5, 1.0, 0.0)  # Highlight not forest

    # Stack masks along the third dimension
    stacked_masks = tf.concat([mask_forest, mask_not_forest], axis=-1)
    
    return image, stacked_masks

def preprocess():
    '''
    Preprocess dataset and split the dataset into train and test.
    '''
    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'meta_data.csv'))
    train_metadata, val_metadata = train_test_split(metadata_df, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        [os.path.join(DATA_DIR, 'images', image_name) for image_name in train_metadata['image']],
        [os.path.join(DATA_DIR, 'masks', mask_name) for mask_name in train_metadata['mask']]
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        [os.path.join(DATA_DIR, 'images', image_name) for image_name in val_metadata['image']],
        [os.path.join(DATA_DIR, 'masks', mask_name) for mask_name in val_metadata['mask']]
    ))

    target_size = (256, 256)
    train_dataset = train_dataset.map(lambda x, y: preprocess_helper(x, y, target_size))
    val_dataset = val_dataset.map(lambda x, y: preprocess_helper(x, y, target_size))

    batch_size = 5

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset