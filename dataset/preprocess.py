import os
import shutil

def copy_jpeg_png_files(source_directory, jpegdest, pngdest):
    
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
                        print(f"Copied: {source_file_path} -> {destination_file_path}")

                    elif filename.endswith("seg.png"):
                        source_file_path = os.path.join(dirpath, filename)
                        
                        relative_dir_path = os.path.relpath(dirpath, source_directory)
                        destination_file_path = os.path.join(pngdest, filename)
                        
                        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                        shutil.copyfile(source_file_path, destination_file_path)
                        print(f"Copied: {source_file_path} -> {destination_file_path}")
                        

current_file_path = os.path.realpath(__file__)


source_directory = os.path.dirname(current_file_path)

print("Source directory:", source_directory)

source_directory_path_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/validation'
jpegdest_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/jpegdest/validation'
pngdest_training = f'{source_directory}/ADE20K_2021_17_01/images/ADE/pngdest/validation'
copy_jpeg_png_files(source_directory_path_training, jpegdest_training, pngdest_training)