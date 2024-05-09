import numpy as np
import warnings
import tensorflow as tf

from model.deeplab import myDeeplab
from model.discriminator import FCDiscriminator
from dataset.preprocess import preprocess
from utils import reload_pretrained
from utils.visualizer import plot_metric_loss, visualize_saved_results
from dataset.preprocess import preprocess

warnings.filterwarnings("ignore")

def train():
    final_confidence_map = None
    last_images = None
    last_labels = None
    train_dataset, val_dataset = preprocess()

    optimizer_G = tf.keras.optimizers.Adam(learning_rate=0.00125)
    optimizer_D = tf.keras.optimizers.SGD(learning_rate=0.001)
    
    deeplab = myDeeplab(((256, 256, 3)))
    reload_pretrained.restore_model_from_checkpoint('model/pretrained/deeplab_resnet.ckpt', deeplab)
    deeplab.trainables = True
    
    discriminator = FCDiscriminator()
    epoch = 1

    loss_D_list = []
    loss_G_list = []
    iou_D_list = []
    iou_G_list = []
    for batch in train_dataset:
        images, labels = batch
        
        with tf.GradientTape() as tape_G:
            batch_confidence_map = deeplab(images, training=True)            
            D_fake = discriminator(batch_confidence_map)
            
            loss_G_adv = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
            loss_ce = tf.keras.losses.BinaryCrossentropy()(labels, batch_confidence_map)
            loss_G = loss_ce + 0.1 * loss_G_adv
        
        gradients_G = tape_G.gradient(loss_G, deeplab.trainable_variables)
        optimizer_G.apply_gradients(zip(gradients_G, deeplab.trainable_variables))
        
        with tf.GradientTape() as tape_D:
            D_fake = discriminator(batch_confidence_map, training = True)
            D_real = discriminator(labels, training = True)
            
            # Calculate adversarial loss for discriminator
            loss_D_fake = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
            loss_D_real = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(D_real), D_real)
            loss_D = (loss_D_fake + loss_D_real) / 2.0
        
        gradients_D = tape_D.gradient(loss_D, discriminator.trainable_variables)
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
    
    print("======= TESTING =======")
    for batch in val_dataset:
        images, labels = batch
        batch_confidence_map = deeplab(images, training=True)

        last_images = images
        last_labels = labels
        final_confidence_map = batch_confidence_map

        D_fake = discriminator(batch_confidence_map)
        
        loss_G_adv = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
        loss_ce = tf.keras.losses.BinaryCrossentropy()(labels, batch_confidence_map)

        loss_G = loss_ce + 0.1 * loss_G_adv
        
        D_fake = discriminator(batch_confidence_map, training = True)
        D_real = discriminator(labels, training = True)
        
        loss_D_fake = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
        loss_D_real = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(D_real), D_real)
        loss_D = (loss_D_fake + loss_D_real) / 2.0
        
        D_real_class = np.where(D_real <= 0.5, 0, 1)
        D_fake_class = np.where(D_fake <= 0.5, 0, 1)
        
        iou_G = tf.keras.metrics.MeanIoU(num_classes = 2)(labels, np.where(batch_confidence_map <= 0.5, 0, 1))
        iou_D = tf.keras.metrics.MeanIoU(num_classes = 2)(D_real_class, D_fake_class)
        print(f"epoch: {epoch} loss_G: {loss_G.numpy():.6f}; iou_G: {iou_G.numpy()*100:.6f}; loss_D: {loss_D.numpy():.6f}; iou_D: {iou_D.numpy()*100:.6f}")
        epoch += 1
    
    np.save('last_images.npy', last_images)
    np.save('last_labels.npy', last_labels)
    np.save('final_confidence_map.npy', final_confidence_map)
    
    visualize_saved_results()
    
if __name__ == '__main__':
    train()