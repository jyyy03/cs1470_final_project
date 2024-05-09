import numpy as np
from parser.parser import TrainArgParser
from model.new_deeplab import myDeeplab
from model.discriminator import FCDiscriminator
from dataset.preprocess import preprocess
import timeit
import warnings

from utils import reload_pretrained
from utils.plot import plot_metric_loss
import tensorflow as tf
from dataset.preprocess import preprocess
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
start = timeit.default_timer()

def train(args):
    final_confidence_map = None
    last_images = None
    last_labels = None
    train_dataset, val_dataset = preprocess()

    optimizer_G = tf.keras.optimizers.Adam(learning_rate=0.00125)
    optimizer_D = tf.keras.optimizers.SGD(learning_rate=0.001)
    
    deeplab = myDeeplab(((256, 256, 3)))
    reload_pretrained.restore_model_from_checkpoint('model/pretrained/deeplab_resnet.ckpt', deeplab)
    deeplab.trainables = True
    
    discriminator = FCDiscriminator(num_classes=2)
    epoch = 1

    loss_D_list = []
    loss_G_list = []
    iou_D_list = []
    iou_G_list = []
    for batch in train_dataset:
        images, labels = batch
        last_images = images
        last_labels = labels
        
        with tf.GradientTape() as tape_G:
            batch_confidence_map = deeplab(images, training=True)
            final_confidence_map = batch_confidence_map
            
            # Forward pass through discriminator
            D_fake = discriminator(batch_confidence_map)
            
            # Calculate adversarial loss for generator (Deeplab)
            loss_G_adv = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
            
            # Calculate Cross-Entropy loss for generator (Deeplab)
            loss_ce = tf.keras.losses.BinaryCrossentropy()(labels, batch_confidence_map)
            
            # Combine adversarial loss and Cross-Entropy loss for generator
            # TODO Change this 0.1 to lamda g_adv
            loss_G = loss_ce + 0.1 * loss_G_adv
        
        # Calculate gradients for generator
        gradients_G = tape_G.gradient(loss_G, deeplab.trainable_variables)
        
        # Update generator
        optimizer_G.apply_gradients(zip(gradients_G, deeplab.trainable_variables))
        
        with tf.GradientTape() as tape_D:
            # Forward pass through discriminator
            D_fake = discriminator(batch_confidence_map, training = True)
            D_real = discriminator(labels, training = True)
            
            # Calculate adversarial loss for discriminator
            loss_D_fake = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
            loss_D_real = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(D_real), D_real)
            loss_D = (loss_D_fake + loss_D_real) / 2.0
        
        # Calculate gradients for discriminator
        gradients_D = tape_D.gradient(loss_D, discriminator.trainable_variables)
        
        # Update discriminator
        optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))
        D_real_class = np.where(D_real <= 0.5, 0, 1)
        D_fake_class = np.where(D_fake <= 0.5, 0, 1)
        
        iou_G = tf.keras.metrics.MeanIoU(num_classes = 2)(labels, np.where(batch_confidence_map <= 0.5, 0, 1))
        iou_D = tf.keras.metrics.MeanIoU(num_classes = 2)(D_real_class, D_fake_class)
        
        loss_D_list.append(loss_D.numpy())
        loss_G_list.append(loss_G.numpy())
        iou_G_list.append(iou_G.numpy())
        iou_D_list.append(iou_D.numpy())
        
        print(f"epoch: {epoch} loss_G: {loss_G.numpy():.6f}; iou_G: {iou_G.numpy()*100:.6f}; loss_D: {loss_D.numpy():.6f}; iou_D: {iou_D.numpy()*100:.6f}")
        epoch += 1
        
    plot_metric_loss(epoch, loss_D_list, loss_G_list, iou_G_list, iou_D_list)

    
    np.save('last_images.npy', last_images)
    np.save('last_labels.npy', last_labels)
    np.save('final_confidence_map.npy', final_confidence_map)
    
    # Testing
    print("======= TESTING =======")
    for batch in val_dataset:
        images, labels = batch
        last_images = images
        last_labels = labels
        
        
        batch_confidence_map = deeplab(images, training=True)
        final_confidence_map = batch_confidence_map
        
        # Forward pass through discriminator
        D_fake = discriminator(batch_confidence_map)
        
        # Calculate adversarial loss for generator (Deeplab)
        loss_G_adv = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
        
        # Calculate Cross-Entropy loss for generator (Deeplab)
        loss_ce = tf.keras.losses.BinaryCrossentropy()(labels, batch_confidence_map)
        
        # Combine adversarial loss and Cross-Entropy loss for generator
        # TODO Change this 0.1 to lamda g_adv
        loss_G = loss_ce + 0.1 * loss_G_adv
        
        
        # Forward pass through discriminator
        D_fake = discriminator(batch_confidence_map, training = True)
        D_real = discriminator(labels, training = True)
        
        # Calculate adversarial loss for discriminator
        loss_D_fake = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
        loss_D_real = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(D_real), D_real)
        loss_D = (loss_D_fake + loss_D_real) / 2.0
        
        D_real_class = np.where(D_real <= 0.5, 0, 1)
        D_fake_class = np.where(D_fake <= 0.5, 0, 1)
        
        iou_G = tf.keras.metrics.MeanIoU(num_classes = 2)(labels, np.where(batch_confidence_map <= 0.5, 0, 1))
        iou_D = tf.keras.metrics.MeanIoU(num_classes = 2)(D_real_class, D_fake_class)
        print(f"epoch: {epoch} loss_G: {loss_G.numpy():.6f}; iou_G: {iou_G.numpy()*100:.6f}; loss_D: {loss_D.numpy():.6f}; iou_D: {iou_D.numpy()*100:.6f}")
        epoch += 1
    
    
    visualize_saved_results()
    
    # np.save('last_images.npy', last_images)
    # np.save('last_labels.npy', last_labels)
    # np.save('final_confidence_map.npy', final_confidence_map)
    # visualize_saved_results()

'''
This can be called in main after after training to visualize saved results from the final batch
'''
def visualize_saved_results():
    # Change this number to view a different set of 5 images, labels, and confidence maps
    sample_num = 5
    last_images = np.load('last_images.npy')
    print(last_images.shape)
    last_labels = np.load('last_labels.npy')
    print(last_labels.shape)
    final_confidence_map = np.load('final_confidence_map.npy')
    print(final_confidence_map.shape)
    final_confidence_map_threshold = np.where(final_confidence_map > 0.5, 0,1)

    # visualize_helper(last_images[5*sample_num:5+5*sample_num], last_labels[5*sample_num:5+5*sample_num], final_confidence_map_threshold[5*sample_num:5+5*sample_num])
    visualize_helper(last_images[:sample_num-1], last_labels[:sample_num-1], final_confidence_map_threshold[:sample_num-1])

'''
Generates a plot with images, ground truth segmentations, and confidence maps for 5 samples
'''
def visualize_helper(images, labels, confidence_maps):
    print(images.shape)
    num_samples = 3
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(num_samples):
        plt.subplot(num_samples, 5, 5*i + 1)
        plt.imshow(images[i])
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_samples, 5, 5*i + 2)
        plt.imshow(labels[i, :, :, 0], cmap='plasma')
        plt.title('Ground Truth 1')
        plt.axis('off')

        plt.subplot(num_samples, 5, 5*i + 3)
        plt.imshow(labels[i, :, :, 1], cmap='plasma')
        plt.title('Ground Truth 2')
        plt.axis('off')

        plt.subplot(num_samples, 5, 5*i + 4)
        plt.imshow(confidence_maps[i, :, :, 0], cmap='plasma')
        print("pred 1: ", confidence_maps[i, :, :, 0])
        plt.title('Class 1 Confidence Map')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(num_samples, 5, 5*i + 5)
        plt.imshow(confidence_maps[i, :, :, 1], cmap='plasma')
        print("pred 2: ", confidence_maps[i, :, :, 1])
        plt.title('Class 2 Confidence Map')
        plt.colorbar()
        plt.axis('off')
    plt.show()

def main(args):    
    train(args)
    # visualize_saved_results()

if __name__ == '__main__':
    train_parser = TrainArgParser()
    main(train_parser.get_arguments())