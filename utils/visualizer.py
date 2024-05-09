import matplotlib.pyplot as plt
import numpy as np

def plot_metric_loss(epoch, loss_D, loss_G, iou_G, iou_D):
    '''
    Generates a plot for loss and metric per epoch and stores the plot locally.
    '''
    epochs = list(range(1, epoch)) 
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_G, label='Segmentation Network\'s Loss')
    plt.plot(epochs, loss_D, label='Discriminator Network\'s Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x * 100 for x in iou_G], label='Segmentation Network\'s Mean IoU')
    plt.plot(epochs, [x * 100 for x in iou_D], label='Discriminator Network\'s Mean IoU')
    plt.title('Mean IoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig('loss_metric.png', dpi=300, transparent=True)

def visualize_dataset(dataset, num_samples=2):
    plt.figure(figsize=(15, num_samples * 5))
    for i, (image, mask) in enumerate(dataset.take(num_samples)):
        
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(mask[:,:,0])  
        plt.title('First mask')
        plt.axis('off')

        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(mask[:,:,-1])  
        plt.title('Second mask')
        plt.axis('off')
    plt.show()


def visualize_saved_results(sample_num=2):
    '''
    This can be called in main after after training to visualize saved results from the final batch
    '''
    last_images = np.load('last_images.npy')
    last_labels = np.load('last_labels.npy')
    final_confidence_map = np.load('final_confidence_map.npy')
    final_confidence_map_threshold = np.where(final_confidence_map > 0.5, 0,1)

    visualize_helper(last_images[:sample_num], last_labels[:sample_num], final_confidence_map_threshold[:sample_num])


def visualize_helper(images, labels, confidence_maps, num_samples = 2):
    '''
    Generates a plot with images, ground truth segmentations, and confidence maps for 5 samples
    '''    
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