import os
import shutil

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
                        destination_file_path = os.path.join(pngdest, filename)
                        
                        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                        shutil.copyfile(source_file_path, destination_file_path)
                        # print(f"Copied: {source_file_path} -> {destination_file_path}")
                        

def preprocess():
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

if __name__ == '__main__':
    preprocess()
