import os
import shutil
import tensorflow as tf

def copy_jpeg_png_files(source_directory, jpegdest, pngdest):
    print(f"======= Start preprocessing data for {source_directory.split('/')[-1]}... ========")
    if not os.path.exists(jpegdest):
        os.makedirs(jpegdest)

    if not os.path.exists(pngdest):
        os.makedirs(pngdest)

    for root, dirs, files in os.walk(source_directory):
        for dirname in dirs:
            subdir_path = os.path.join(root, dirname)
            for dirpath, dirnames, filenames in os.walk(subdir_path):
                for filename in filenames:
                    if filename.endswith(".jpg"):
                        source_file_path = os.path.join(dirpath, filename)
                        
                        destination_file_path = os.path.join(jpegdest, filename)
                        
                        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                        shutil.copyfile(source_file_path, destination_file_path)
                        # print(f"Copied: {source_file_path} -> {destination_file_path}")

                    elif filename.endswith("seg.png"):
                        source_file_path = os.path.join(dirpath, filename)
                        
                        relative_dir_path = os.path.relpath(dirpath, source_directory)
                        destination_file_path = os.path.join(pngdest, filename.replace('_seg', ''))
                        
                        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                        shutil.copyfile(source_file_path, destination_file_path)
                        # print(f"Copied: {source_file_path} -> {destination_file_path}")
                        

def copy_files():
    current_file_path = os.path.realpath(__file__)
    source_directory = os.path.dirname(current_file_path)

    source_directory_path_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/training'
    jpegdest_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/jpegdest/training'
    pngdest_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/pngdest/training'

    source_directory_path_val = f'{source_directory}/ADE20K_2021_17_01/images/ADE/validation'
    jpegdest_val = f'{source_directory}/ADE20K_2021_17_01/images/ADE/jpegdest/validation'
    pngdest_val = f'{source_directory}/ADE20K_2021_17_01/images/ADE/pngdest/validation'

    copy_jpeg_png_files(source_directory_path_training, jpegdest_training, pngdest_training)
    copy_jpeg_png_files(source_directory_path_val, jpegdest_val, pngdest_val)


def load_image_label(image_path, label_path):
    # Load the image file
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # for RGB images
    image = tf.image.resize(image, [321, 321]) 
    image = image / 255.0 

    # Load the label file
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=3)  # assuming label is a grayscale image
    label = tf.image.resize(label, [321, 321], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.cast(label, dtype=tf.float32)
    label = label / 255.0
    return image, label

def preprocess(is_training = True):
    current_file_path = os.path.realpath(__file__)
    data_dir = os.path.dirname(current_file_path)
    purpose = 'training' if is_training else 'validation'
    image_dir = os.path.join(data_dir, f'ADE20K_2021_17_01/images/ADE/jpegdest/{purpose}')
    label_dir = os.path.join(data_dir, f'ADE20K_2021_17_01/images/ADE/pngdest/{purpose}')

    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    image_paths = [os.path.join(image_dir, f) for f in image_files if f.endswith('.jpg')]
    label_paths = [os.path.join(label_dir, f) for f in label_files if f.endswith('.png')]

    image_paths.sort()
    label_paths.sort()

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.map(load_image_label)

    batch_size = 1024
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    copy_files()