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


# 5108 Unique Values
# 4108 Training
# 1000 Validation



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
    data_dir = 'Forest/Forest Segmented/Forest Segmented'

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
    target_size = (321, 321)
    train_dataset = train_dataset.map(lambda x, y: preprocess_helper(x, y, target_size))
    val_dataset = val_dataset.map(lambda x, y: preprocess_helper(x, y, target_size))

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
# Visualize samples from the training dataset
visualize_dataset(train_dataset)

# Visualize samples from the validation dataset
visualize_dataset(val_dataset)




# def copy_jpeg_png_files(source_directory, jpegdest, pngdest):
#     print(f"======= Start preprocessing data for {source_directory.split('/')[-1]}... ========")
#     if not os.path.exists(jpegdest):
#         os.makedirs(jpegdest)

#     if not os.path.exists(pngdest):
#         os.makedirs(pngdest)

#     for root, dirs, files in os.walk(source_directory):
#         for dirname in dirs:
#             subdir_path = os.path.join(root, dirname)
#             for dirpath, dirnames, filenames in os.walk(subdir_path):
#                 for filename in filenames:
#                     if filename.endswith(".jpg"):
#                         source_file_path = os.path.join(dirpath, filename)
                        
#                         destination_file_path = os.path.join(jpegdest, filename)
                        
#                         os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
#                         shutil.copyfile(source_file_path, destination_file_path)
#                         # print(f"Copied: {source_file_path} -> {destination_file_path}")

#                     elif filename.endswith("seg.png"):
#                         source_file_path = os.path.join(dirpath, filename)
                        
#                         relative_dir_path = os.path.relpath(dirpath, source_directory)
#                         destination_file_path = os.path.join(pngdest, filename.replace('_seg', ''))
                        
#                         os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
#                         shutil.copyfile(source_file_path, destination_file_path)
#                         # print(f"Copied: {source_file_path} -> {destination_file_path}")
                        

# def copy_files():
#     current_file_path = os.path.realpath(__file__)
#     source_directory = os.path.dirname(current_file_path)

#     source_directory_path_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/training'
#     jpegdest_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/jpegdest/training'
#     pngdest_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/pngdest/training'

#     source_directory_path_val = f'{source_directory}/ADE20K_2021_17_01/images/ADE/validation'
#     jpegdest_val = f'{source_directory}/ADE20K_2021_17_01/images/ADE/jpegdest/validation'
#     pngdest_val = f'{source_directory}/ADE20K_2021_17_01/images/ADE/pngdest/validation'

#     copy_jpeg_png_files(source_directory_path_training, jpegdest_training, pngdest_training)
#     copy_jpeg_png_files(source_directory_path_val, jpegdest_val, pngdest_val)


# def load_image_label(image_path, label_path):
#     # Load the image file
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)  # for RGB images
#     image = tf.image.resize(image, [321, 321]) 
#     image = image / 255.0 

#     # Load the label file
#     label = tf.io.read_file(label_path)
#     label = tf.image.decode_png(label, channels=3)  # assuming label is a grayscale image
#     label = tf.image.resize(label, [321, 321], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     label = tf.cast(label, dtype=tf.float32)
#     label = label / 255.0
#     return image, label

# def preprocess(is_training = True):
#     current_file_path = os.path.realpath(__file__)
#     data_dir = os.path.dirname(current_file_path)
#     purpose = 'training' if is_training else 'validation'
#     image_dir = os.path.join(data_dir, f'ADE20K_2021_17_01/images/ADE/jpegdest/{purpose}')
#     label_dir = os.path.join(data_dir, f'ADE20K_2021_17_01/images/ADE/pngdest/{purpose}')

#     image_files = os.listdir(image_dir)
#     label_files = os.listdir(label_dir)

#     image_paths = [os.path.join(image_dir, f) for f in image_files if f.endswith('.jpg')]
#     label_paths = [os.path.join(label_dir, f) for f in label_files if f.endswith('.png')]

#     image_paths.sort()
#     label_paths.sort()

#     dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
#     dataset = dataset.map(load_image_label)

#     batch_size = 10
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return dataset

# if __name__ == '__main__':
#     copy_files()