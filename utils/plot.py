import matplotlib.pyplot as plt

def plot_metric_loss(epoch, loss_D, loss_G, iou_G, iou_D):
    epochs = list(range(1, epoch)) 
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_G, label='Generator Loss')
    plt.plot(epochs, loss_D, label='Discriminator Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x * 100 for x in iou_G], label='Generator Mean IoU')
    plt.plot(epochs, [x * 100 for x in iou_D], label='Discriminator Mean IoU')
    plt.title('Mean IoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig('loss_metric.png', dpi=300, transparent=True)
