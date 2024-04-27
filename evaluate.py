from parser.parser import EvalArgParser

def main(args):
    print(args)

if __name__ == '__main__':
    eval_parser = EvalArgParser()
    main(eval_parser.get_arguments())

# import argparse
# import os
# import numpy as np
# from PIL import Image
# from multiprocessing import Pool
# from collections import OrderedDict
# from packaging import version
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.utils import get_file

# from model.deeplab import Res_Deeplab
# from dataset.voc_dataset import VOCDataSet

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# pretrained_models_dict = {
#     'semi0.125': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-03c6f81c.pth',
#     'semi0.25': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.25-473f8a14.pth',
#     'semi0.5': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.5-acf6a654.pth',
#     'advFull': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSegVOCFull-92fbc7ee.pth'
# }

# def get_arguments():
#     """Parse all the arguments provided from the CLI.
    
#     Returns:
#         A list of parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description="VOC evaluation script")
#     parser.add_argument("--model", type=str, default='DeepLab',
#                         help="Available options: DeepLab/DRN")
#     parser.add_argument("--data-dir", type=str, default='./dataset/VOC2012',
#                         help="Path to the directory containing the PASCAL VOC dataset.")
#     parser.add_argument("--data-list", type=str, default='./dataset/voc_list/val.txt',
#                         help="Path to the file listing the images in the dataset.")
#     parser.add_argument("--ignore-label", type=int, default=255,
#                         help="The index of the label to ignore during the training.")
#     parser.add_argument("--num-classes", type=int, default=21,
#                         help="Number of classes to predict (including background).")
#     parser.add_argument("--restore-from", type=str,
#                         default='http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-8d75b3f1.pth',
#                         help="Where to restore model parameters from.")
#     parser.add_argument("--pretrained-model", type=str, default=None,
#                         help="Where to restore model parameters from.")
#     parser.add_argument("--save-dir", type=str, default='results',
#                         help="Directory to store results.")
#     parser.add_argument("--gpu", type=int, default=0,
#                         help="Choose GPU device.")
#     return parser.parse_args()

# class VOCColorize:
#     def __init__(self, n=22):
#         self.cmap = color_map(22)
#         self.cmap = tf.convert_to_tensor(self.cmap[:n])

#     def __call__(self, gray_image):
#         size = gray_image.shape
#         color_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)

#         for label in range(len(self.cmap)):
#             mask = (label == gray_image)
#             color_image[mask] = self.cmap[label]

#         # handle void
#         mask = (255 == gray_image)
#         color_image[mask] = [255, 255, 255]

#         return color_image

# def color_map(N=256, normalized=False):
#     def bitget(byteval, idx):
#         return ((byteval & (1 << idx)) != 0)

#     dtype = 'float32' if normalized else 'uint8'
#     cmap = np.zeros((N, 3), dtype=dtype)
#     for i in range(N):
#         r = g = b = 0
#         c = i
#         for j in range(8):
#             r = r | (bitget(c, 0) << 7-j)
#             g = g | (bitget(c, 1) << 7-j)
#             b = b | (bitget(c, 2) << 7-j)
#             c = c >> 3

#         cmap[i] = np.array([r, g, b])

#     cmap = cmap / 255 if normalized else cmap
#     return cmap

# def get_iou(data_list, class_num, save_path=None):
#     from utils.metric import ConfusionMatrix

#     ConfM = ConfusionMatrix(class_num)
#     f = ConfM.generateM
#     with Pool() as pool:
#         m_list = pool.map(f, data_list)

#     for m in m_list:
#         ConfM.addM(m)

#     aveJ, j_list, M = ConfM.jaccard()

#     classes = np.array(('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'))

#     for i, iou in enumerate(j_list):
#         print(f'class {i:2d} {classes[i]:12} IU {j_list[i]:.2f}')

#     print('meanIOU:', aveJ)
#     if save_path:
#         with open(save_path, 'w') as f:
#             for i, iou in enumerate(j_list):
#                 f.write(f'class {i:2d} {classes[i]:12} IU {j_list[i]:.2f}\n')
#             f.write(f'meanIOU: {aveJ}\n')

# def show_all(gt, pred):
#     import matplotlib.pyplot as plt
#     from matplotlib import colors
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

#     fig, axes = plt.subplots(1, 2)
#     ax1, ax2 = axes

#     classes = np.array(('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'))
#     colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5), (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), (0, 0.25, 0.5)]
#     cmap = colors.ListedColormap(colormap)
#     bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
#     norm = colors.BoundaryNorm(bounds, cmap.N)

#     ax1.set_title('gt')
#     ax1.imshow(gt, cmap=cmap, norm=norm)

#     ax2.set_title('pred')
#     ax2.imshow(pred, cmap=cmap, norm=norm)

#     plt.show()

# def main():
#     """Create the model and start the evaluation process."""
#     args = get_arguments()

#     gpu0 = args.gpu

#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)

#     model = Res_Deeplab(num_classes=args.num_classes)

#     if args.pretrained_model is not None:
#         args.restore_from = pretrained_models_dict[args.pretrained_model]

#     if args.restore_from[:4] == 'http':
#         weights_path = get_file(args.restore_from.split('/')[-1], args.restore_from)
#         saved_state_dict = tf.keras.utils.get_file(args.restore_from, weights_path)
#     else:
#         saved_state_dict = tf.keras.models.load_model(args.restore_from)
#     model.load_weights(saved_state_dict)

#     model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

#     testloader = tf.data.Dataset.from_tensor_slices(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False)).batch(1)

#     data_list = []
#     colorize = VOCColorize()

#     for index, (image, label, size, name) in enumerate(testloader):
#         if index % 100 == 0:
#             print(f'{index} processed')

#         output = model(image, training=False)
#         output = tf.image.resize(output, size=(505, 505), method='bilinear')
#         output = tf.argmax(output, axis=-1).numpy()[0]

#         output = output[:size[0], :size[1]]
#         gt = label[0].numpy()[:size[0], :size[1]].astype(np.int)

#         filename = os.path.join(args.save_dir, f'{name[0]}.png')
#         color_file = Image.fromarray(colorize(output).astype(np.uint8), 'RGB')
#         color_file.save(filename)

#         data_list.append([gt.flatten(), output.flatten()])

#     filename = os.path.join(args.save_dir, 'result.txt')
#     get_iou(data_list, args.num_classes, filename)

# if __name__ == '__main__':
#     main()
