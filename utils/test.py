import torchvision
import tensorflow as tf

def make_grid(images, nrow=8, padding=2, normalize=False, range_=None, scale_each=False, pad_value=0):
    """Make a grid of images."""
    # Normalize the images if required
    if normalize:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        if range_ is not None:
            images = tf.clip_by_value(images, range_[0], range_[1])
    # Scale each image individually if required
    if scale_each:
        images_min = tf.reduce_min(images, axis=[1, 2], keepdims=True)
        images_max = tf.reduce_max(images, axis=[1, 2], keepdims=True)
        images = (images - images_min) / (images_max - images_min)
    # Add padding if required
    if padding > 0:
        images = tf.pad(images, [[0, 0], [padding, padding], [padding, padding], [0, 0]], constant_values=pad_value)
    # Create the grid
    batch_size, height, width, channels = images.shape
    ncol = (batch_size + nrow - 1) // nrow
    grid_height = height * ncol + padding * (ncol - 1)
    grid_width = width * nrow + padding * (nrow - 1)
    grid = tf.fill([grid_height, grid_width, channels], pad_value)
    for i, image in enumerate(images):
        y = (i // nrow) * (height + padding)
        x = (i % nrow) * (width + padding)
        grid[y:y+height, x:x+width].assign(image)
    return grid

print(torchvision.utils.make_grid(imgs).numpy())