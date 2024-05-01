import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt



# 100 x height x width x 2

# batch[0] is image: batchsize x height x width x 3
# -> generated: batchsize x height x width x 2 (Nora)

# batch[1] is masks: 10 x height x width x 2



import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd


# Function to preprocess image and generate two masks
def preprocess_helper(image_path, mask_path, target_size=(321, 321)):
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    
    # Read mask
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
    # Path to data directory
    data_dir = 'dataset/Forest/Forest Segmented/Forest Segmented'

    # Load metadata CSV
    metadata_df = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))

    # Split data into training and validation sets
    train_metadata, val_metadata = train_test_split(metadata_df, test_size=0.2, random_state=42)

    # Create TensorFlow dataset for training and validation data
    train_dataset = tf.data.Dataset.from_tensor_slices((
        [os.path.join(data_dir, 'images', image_name) for image_name in train_metadata['image']],
        [os.path.join(data_dir, 'masks', mask_name) for mask_name in train_metadata['mask']]
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        [os.path.join(data_dir, 'images', image_name) for image_name in val_metadata['image']],
        [os.path.join(data_dir, 'masks', mask_name) for mask_name in val_metadata['mask']]
    ))

    # Preprocess data
    target_size = (256, 256)
    train_dataset = train_dataset.map(lambda x, y: preprocess_helper(x, y, target_size))
    val_dataset = val_dataset.map(lambda x, y: preprocess_helper(x, y, target_size))

    batch_size = 20
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Now train_dataset and val_dataset contain preprocessed data ready for training your segmentation model
    # print(train_dataset)
    # print(val_dataset)
    return train_dataset, val_dataset

def visualize_dataset(dataset, num_samples=5):
    plt.figure(figsize=(15, num_samples * 5))
    for i, (image, mask) in enumerate(dataset.take(num_samples)):
        
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(mask[:,:,0])  # Assuming single-channel mask
        plt.title('First mask')
        plt.axis('off')

        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(mask[:,:,-1])  # Assuming single-channel mask
        plt.title('Second mask')
        plt.axis('off')
    plt.show()

train_dataset, val_dataset = preprocess()