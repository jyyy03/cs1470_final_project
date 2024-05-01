import numpy as np
from parser.parser import TrainArgParser
from model.new_deeplab import myDeeplab
from model.discriminator import FCDiscriminator
from dataset.preprocess import preprocess
import timeit
import warnings

from utils import train_utils, reload_pretrained
import tensorflow as tf
from dataset.preprocess import preprocess
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
start = timeit.default_timer()
def train_one_step(model, model_D, interp, trainloader_iter, trainloader_gt_iter, trainloader_remain_iter, step, args):
    """
    Trains the model with one epoch.
    """
    # loss_seg_value = 0
    # loss_adv_pred_value = 0
    # loss_D_value = 0
    # loss_semi_value = 0
    # loss_semi_adv_value = 0
    # optimizer_callback = tf.keras.callbacks.LearningRateScheduler(train_utils.adjust_learning_rate)
    # # 1. train semi
    # for sub_i in range(args.iter_size):
    #     if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and step >= args.semi_start_adv:
    #         _, batch = trainloader_remain_iter.get_next()
    #         images = batch[0]
    #         images = tf.convert_to_tensor(images, dtype=tf.float32)

    #         pred = interp(model(images))
    #         D_out = interp(model_D(tf.nn.softmax(pred)))
    #         D_out_sigmoid = tf.nn.sigmoid(D_out).numpy().squeeze(axis=1)

    #         ignore_mask_remain = np.zeros(D_out_sigmoid.shape, dtype=np.bool)
    #         loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
    #         loss_semi_adv /= args.iter_size
    #         loss_semi_adv_value += loss_semi_adv.numpy() / args.lambda_semi_adva

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
        print(f"epoch: {epoch} loss_G: {loss_G.numpy():.6f}; iou_G: {iou_G.numpy()*100:.6f}; loss_D: {loss_D.numpy():.6f}; iou_D: {iou_D.numpy()*100:.6f}")
        epoch += 1
    
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


# def train(args):
#     final_confidence_map = None
#     last_images = None
#     last_labels = None
#     train_dataset, val_dataset = preprocess()
#     print("===== preprocess complete ======")
    
#     optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00125)
#     deeplab = myDeeplab(((256, 256, 3)))
#     reload_pretrained.restore_model_from_checkpoint('model/pretrained/deeplab_resnet.ckpt', deeplab)
#     deeplab.trainables = True
#     print('====== Start feeding deeplab ===== ')
#     for batch in train_dataset:
#         images, labels = batch
#         # print("image", images[0])
#         # print("ground truth 1: ", labels[0, :, :, 0])
#         # print("ground truth 2: ", labels[0, :, :, 1])
#         last_images = images
#         last_labels = labels
        
#         with tf.GradientTape() as tape:
#             batch_confidence_map = deeplab(images, training=True)
#             final_confidence_map = batch_confidence_map
#             # print("pred 1: ", batch_confidence_map[0, :, :, 0])
#             # print("pred 2: ", batch_confidence_map[0, :, :, 1])
#             loss_ce = tf.keras.losses.BinaryCrossentropy()(batch_confidence_map, labels)
#             print(f'====== Loss is {loss_ce.numpy()} ===== ')
        
#         gradients = tape.gradient(loss_ce, deeplab.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, deeplab.trainable_variables))
        
#     print('====== Deeplab Complete===== ')

#     # Save the final results from training
#     np.save('last_images.npy', last_images)
#     np.save('last_labels.npy', last_labels)
#     np.save('final_confidence_map.npy', final_confidence_map)

#     # visualize with final_confidence_map here
#     visualize_saved_results()

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
    visualize_helper(last_images[5*sample_num:5+5*sample_num], last_labels[5*sample_num:5+5*sample_num], final_confidence_map_threshold[5*sample_num:5+5*sample_num])
    # visualize_helper(last_images[:sample_num], last_labels[:sample_num], final_confidence_map_threshold[:sample_num])
    # visualize_helper(last_images[0:5], last_labels[0:5], final_confidence_map_threshold[0:5])

'''
Generates a plot with images, ground truth segmentations, and confidence maps for 5 samples
'''
def visualize_helper(images, labels, confidence_maps):
    print(images.shape)
    num_samples = 5
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


    # trainloader, trainloader_gt, trainloader_remain = train_utils.load_ade20(args) # TODO: wait for dataset

    # model_Deeplab = Res_Deeplab()
    # model_Dis = FCDiscriminator()
    # train_utils.compile_model(model_Deeplab, model_Dis, args)
    
   
    # # # loss/ bilinear upsampling
    # # bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # # Upsampling layer
    # h, w = map(int, args.input_size.split(','))
    # input_size = (h, w)
    # upsampling = tf.keras.layers.UpSampling2D(size=(input_size[1], input_size[0]), interpolation='bilinear')
        
    
    # # labels for adversarial training
    # pred_label = 0
    # gt_label = 1

    # for i_iter in range(args.num_steps):
    #     loss_seg_value = 0
    #     loss_adv_pred_value = 0
    #     loss_D_value = 0
    #     loss_semi_value = 0
    #     loss_semi_adv_value = 0
    #     adjust_learning_rate(optimizer_S, i_iter)
    #     adjust_learning_rate_D(optimizer_D, i_iter)
    #     for sub_i in range(args.iter_size):
    #         # TODO: don't accumulate grads in D
    #         # for param in model_D.parameters():
    #         #     param.requires_grad = False

    #         # do semi:
    #         if (args.lambda_semi > 0 or args.lambda_semi_adv > 0) and i_iter >= args.semi_start_adv:
    #             try:
    #                 _, batch = next(trainloader_remain_iter)
    #             except StopIteration:
    #                 trainloader_remain_iter = enumerate(trainloader_remain)
    #                 _, batch = next(trainloader_remain_iter)

    #             # Only access to img
    #             images = batch[0]  # Assuming images are the first element of the batch
    #             images = tf.convert_to_tensor(images, dtype=tf.float32)

    #             # Forward pass through the model with interpolation
    #             pred = interp(model(images))
    #             pred_remain = tf.stop_gradient(pred)

    #             D_out = interp(model_D(tf.nn.softmax(pred)))
    #             D_out_sigmoid = tf.nn.sigmoid(D_out).numpy().squeeze(axis=1)

    #             ignore_mask_remain = np.zeros(D_out_sigmoid.shape, dtype=np.bool)
    #             loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
    #             loss_semi_adv /= args.iter_size
    #             loss_semi_adv_value += loss_semi_adv.numpy() / args.lambda_semi_adv

    #             if args.lambda_semi <= 0 or i_iter < args.semi_start:
    #                 loss_semi_value = 0
    #             else:
    #                 semi_ignore_mask = D_out_sigmoid < args.mask_T
    #                 semi_gt = np.argmax(pred.numpy(), axis=1)
    #                 semi_gt[semi_ignore_mask] = 255

    #                 semi_ratio = 1.0 - float(np.sum(semi_ignore_mask)) / semi_ignore_mask.size
    #                 print('semi ratio: {:.4f}'.format(semi_ratio))

    #                 if semi_ratio == 0.0:
    #                     loss_semi_value += 0
    #                 else: 
    #                     semi_gt = tf.convert_to_tensor(semi_gt, dtype=tf.float32)
    #                     loss_semi = args.lambda_semi * loss_calc(pred, semi_gt)
    #                     loss_semi /= args.iter_size
    #                     loss_semi_value += loss_semi.numpy() / args.lambda_semi
    #                     loss_semi += loss_semi_adv
    #         else:
    #             loss_semi = None
    #             loss_semi_adv = None
            
    #         # train with source
    #         try:
    #             _, batch = next(trainloader_iter)
    #         except StopIteration:
    #             trainloader_iter = iter(trainloader)
    #             _, batch = next(trainloader_iter)
    #         images, labels, _, _ = batch
    #         images = tf.Variable(images).cuda(args.gpu) if args.gpu >= 0 else tf.Variable(images)
    #         ignore_mask = (labels.numpy() == 255)
    #         pred = interp(model(images))

    #         loss_seg = loss_calc(pred, labels)

    #         D_out = interp(model_D(tf.nn.softmax(pred)))

    #         loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

    #         loss = loss_seg + args.lambda_adv_pred * loss_adv_pred
    #         # proper normalization
    #         loss /= args.iter_size
    #         loss_seg_value += loss_seg.numpy() / args.iter_size
    #         loss_adv_pred_value += loss_adv_pred.numpy() / args.iter_size

    #         # train D
    #         for param in model_D.trainable_variables:
    #             param.trainable = True

    #         # train with pred
    #         pred = tf.stop_gradient(pred)

    #         if args.D_remain:
    #             pred = tf.concat([pred, pred_remain], axis=0)
    #             ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis=0)

    #         D_out = interp(model_D(tf.nn.softmax(pred)))
    #         loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
    #         loss_D /= args.iter_size * 2

    #         loss_D_value += loss_D.numpy()

    #         # Get ground truth labels
    #         try:
    #             _, batch = next(trainloader_gt_iter)
    #         except StopIteration:
    #             trainloader_gt_iter = iter(trainloader_gt)
    #             _, batch = next(trainloader_gt_iter)

    #         _, labels_gt, _, _ = batch
    #         D_gt_v = tf.Variable(one_hot(labels_gt))
    #         ignore_mask_gt = (labels_gt.numpy() == 255)

    #         # Forward pass through the discriminator model with one-hot encoded ground truth labels
    #         D_out = interp(model_D(D_gt_v))

    #         # Calculate discriminator loss
    #         loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
    #         loss_D /= args.iter_size * 2

    #         # # Compute gradients
    #         # gradients_D = tape.gradient(loss_D, model_D.trainable_variables)

    #         # # Backpropagation
    #         # optimizer_D.apply_gradients(zip(gradients_D, model_D.trainable_variables))

    #         # Accumulate loss_D_value
    #         loss_D_value += loss_D.numpy()

    #     # Print the loss values and other information
    #     print('exp = {}'.format(args.snapshot_dir))
    #     print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value))
    #     if i_iter >= args.num_steps-1:
    #         print('save model ...')
    #         tf.keras.models.save_model(model, osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'.h5'))
    #         tf.keras.models.save_model(model_D, osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'_D.h5'))
    #         break

    #     if i_iter % args.save_pred_every == 0 and i_iter != 0:
    #         print('taking snapshot ...')
    #         tf.keras.models.save_model(model, osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'.h5'))
    #         tf.keras.models.save_model(model_D, osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'_D.h5'))

    #     end = timeit.default_timer()
    #     print(end-start,'seconds')





