from parser.parser import TrainArgParser
# from model.deeplab import Res_Deeplab
from model.new_deeplab import myDeeplab
from model.discriminator import FCDiscriminator
from dataset.preprocess import preprocess
import timeit

from utils import train_utils, reload_pretrained
import tensorflow as tf
from dataset.preprocess import preprocess

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
    train_loader = preprocess()
    print("preprocess complete")
    
    optimizer = tf.keras.optimizers.SGD()
    for batch in train_loader:
        print(f'img {batch[0].shape} \n ======================')
        print(f'label {batch[1].shape}')
        # deeplab = Res_Deeplab(11)
        deeplab = myDeeplab(((batch[0].shape[1], batch[0].shape[2], batch[0].shape[3])))
        reload_pretrained.restore_model_from_checkpoint('model/pretrained/Deeplab Resnet.ckpt', deeplab)
        model_D = FCDiscriminator(11) 
        pred_label = 0
        loss_D_value = 0
        print('====== start feeding deeplab ===== ')
        pred = deeplab.predict_on_batch(batch[0])
        print('====== done!!!!!!!!! ===== ')
        loss_ce = train_utils.loss_function(pred, batch[1])
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            D_out = model_D(tf.nn.softmax(pred))
            loss_D = bce_loss(D_out, train_utils.make_D_label(pred_label, args.ignore_mask))
            loss_D_value += loss_D/args.iter_size/2
        grads = tape.gradient(loss_D, model_D.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_D.trainable_variables))
        print(loss_ce)
        break

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


def main(args):    
    train(args)

if __name__ == '__main__':
    train_parser = TrainArgParser()
    main(train_parser.get_arguments())


