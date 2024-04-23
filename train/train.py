from parser.parser import TrainArgParser
import pickle
import os
import os.path as osp
import pickle


from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
import timeit

import train_utils

start = timeit.default_timer()

def train(args):
    
    model_Deeplab = Res_Deeplab()
    model_Dis = FCDiscriminator
    train_utils.compile_model(model_Deeplab, model_Dis, args)
    trainloader, trainloader_gt, trainloader_remain = train_utils.load_ade20(args)
   
    # # loss/ bilinear upsampling
    # bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # # Upsampling layer
    # if tf.__version__ >= '2.4.0':
    #     interp = tf.keras.layers.UpSampling2D(size=(input_size[1], input_size[0]), interpolation='bilinear', align_corners=True)
    # else:
    #     interp = tf.keras.layers.UpSampling2D(size=(input_size[1], input_size[0]), interpolation='bilinear')
    
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


def main(args):
    train(args)

if __name__ == '__main__':
    train_parser = TrainArgParser()
    main(train_parser.get_arguments())