import tensorflow as tf
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = "/Users/michael/Desktop/CSCI 1470/cs1470_final_project/dataset/ADE20K_2021_17_01/images/ADE/jpegdest",
    image_size = (321, 321)
)


for image_batch, labels_batch in train_ds:
  print(image_batch)
#   print(labels_batch.shape)
  break
