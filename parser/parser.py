import argparse
import numpy as np
import os

upper_level_path = os.path.dirname(os.path.dirname(__file__))
class TrainArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
        self.MODEL = 'DeepLab'
        self.BATCH_SIZE = 10
        self.ITER_SIZE = 1
        self.NUM_WORKERS = 4
        self.DATA_DIRECTORY = f'{upper_level_path}/dataset/ADE20K_2021_17_01'
        self.DATA_LIST_PATH = f'{upper_level_path}/dataset/data/train_aug.txt'
        self.IGNORE_LABEL = 255
        self.INPUT_SIZE = '321,321'
        self.LEARNING_RATE = 2.5e-4
        self.MOMENTUM = 0.9
        self.NUM_CLASSES = 11
        self.NUM_STEPS = 20000
        self.POWER = 0.9
        self.RANDOM_SEED = 1234
        self. RESTORE_FROM = 'model/pretrained/Deeplab Resnet.ckpt'
        self.SAVE_NUM_IMAGES = 2
        self.SAVE_PRED_EVERY = 5000
        # self. SNAPSHOT_DIR = './snapshots/'
        self.WEIGHT_DECAY = 0.0005

        self.LEARNING_RATE_D = 1e-4
        self.LAMBDA_ADV_PRED = 0.1

        self.PARTIAL_DATA=0.5

        self.SEMI_START=5000
        self.LAMBDA_SEMI=0.1
        self.MASK_T=0.2

        self.LAMBDA_SEMI_ADV=0.001
        self.SEMI_START_ADV=0
        self.D_REMAIN=True

    def get_arguments(self):
         """Parse all the arguments provided from the CLI.

        Returns:
        A list of parsed arguments.
        """
         parser = self.parser
         parser.add_argument("--model", type=str, default=self.MODEL,
                            help="available options : DeepLab/DRN")
         parser.add_argument("--batch-size", type=int, default=self.BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
         parser.add_argument("--iter-size", type=int, default=self.ITER_SIZE,
                            help="Accumulate gradients for ITER_SIZE iterations.")
         parser.add_argument("--num-workers", type=int, default=self.NUM_WORKERS,
                            help="number of workers for multithread dataloading.")
         parser.add_argument("--data-dir", type=str, default=self.DATA_DIRECTORY,
                            help="Path to the directory containing the PASCAL VOC dataset.")
         parser.add_argument("--data-list", type=str, default=self.DATA_LIST_PATH,
                            help="Path to the file listing the images in the dataset.")
         parser.add_argument("--partial-data", type=float, default=self.PARTIAL_DATA,
                            help="The index of the label to ignore during the training.")
         parser.add_argument("--partial-id", type=str, default=None,
                            help="restore partial id list")
         parser.add_argument("--ignore-label", type=int, default=self.IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
         parser.add_argument("--input-size", type=str, default=self.INPUT_SIZE,
                            help="Comma-separated string with height and width of images.")
         parser.add_argument("--is-training", action="store_true",
                            help="Whether to updates the running means and variances during the training.")
         parser.add_argument("--learning-rate", type=float, default=self.LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
         parser.add_argument("--learning-rate-D", type=float, default=self.LEARNING_RATE_D,
                            help="Base learning rate for discriminator.")
         parser.add_argument("--lambda-adv-pred", type=float, default=self.LAMBDA_ADV_PRED,
                            help="lambda_adv for adversarial training.")
         parser.add_argument("--lambda-semi", type=float, default=self.LAMBDA_SEMI,
                            help="lambda_semi for adversarial training.")
         parser.add_argument("--lambda-semi-adv", type=float, default=self.LAMBDA_SEMI_ADV,
                             help="lambda_semi for adversarial training.")
         parser.add_argument("--mask-T", type=float, default=self.MASK_T,
                             help="mask T for semi adversarial training.")
         parser.add_argument("--semi-start", type=int, default=self.SEMI_START,
                             help="start semi learning after # iterations")
         parser.add_argument("--semi-start-adv", type=int, default=self.SEMI_START_ADV,
                             help="start semi learning after # iterations")
         parser.add_argument("--D-remain", type=bool, default=self.D_REMAIN,
                             help="Whether to train D with unlabeled data")
         parser.add_argument("--momentum", type=float, default=self.MOMENTUM,
                             help="Momentum component of the optimiser.")
         parser.add_argument("--not-restore-last", action="store_true",
                             help="Whether to not restore last (FC) layers.")
         parser.add_argument("--num-classes", type=int, default=self.NUM_CLASSES,
                             help="Number of classes to predict (including background).")
         parser.add_argument("--num-steps", type=int, default=self.NUM_STEPS,
                             help="Number of training steps.")
         parser.add_argument("--power", type=float, default=self.POWER,
                             help="Decay parameter to compute the learning rate.")
         parser.add_argument("--random-mirror", action="store_true",
                             help="Whether to randomly mirror the inputs during the training.")
         parser.add_argument("--random-scale", action="store_true",
                             help="Whether to randomly scale the inputs during the training.")
         parser.add_argument("--random-seed", type=int, default=self.RANDOM_SEED,
                             help="Random seed to have reproducible results.")
         # parser.add_argument("--restore-from", type=str, default=self.RESTORE_FROM,
         #                     help="Where restore model parameters from.")
         parser.add_argument("--restore-from-D", type=str, default=None,
                             help="Where restore model parameters from.")
         parser.add_argument("--save-num-images", type=int, default=self.SAVE_NUM_IMAGES,
                             help="How many images to save.")
         parser.add_argument("--save-pred-every", type=int, default=self.SAVE_PRED_EVERY,
                             help="Save summaries and checkpoint every often.")
         # parser.add_argument("--snapshot-dir", type=str, default=self.SNAPSHOT_DIR,
         #                     help="Where to save snapshots of the model.")
         parser.add_argument("--weight-decay", type=float, default=self.WEIGHT_DECAY,
                             help="Regularisation parameter for L2-loss.")
         parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
         return parser.parse_args()


class EvalArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="VOC evaluation script")
        self.IMG_MEAN = np.array((123.68748388, 118.66391674, 109.94100899), dtype=np.float32)

        self.MODEL = 'DeepLab'
        self.DATA_DIRECTORY = f'{upper_level_path}/dataset/ADE20K_2021_17_01'
        self.DATA_LIST_PATH = f'{upper_level_path}/dataset/data/validation_aug.txt'
        self.IGNORE_LABEL = 255
        self.NUM_CLASSES = 11
        self.NUM_STEPS = 1449
        # RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-8d75b3f1.pth'
        self.PRETRAINED_MODEL = None
        self.SAVE_DIRECTORY = 'results'


    def get_arguments(self):
        """Parse all the arguments provided from the CLI.

        Returns:
        A list of parsed arguments.
        """
        parser = self.parser
        parser.add_argument("--model", type=str, default=self.MODEL,
                            help="available options : DeepLab/DRN")
        parser.add_argument("--data-dir", type=str, default=self.DATA_DIRECTORY,
                            help="Path to the directory containing the PASCAL VOC dataset.")
        parser.add_argument("--data-list", type=str, default=self.DATA_LIST_PATH,
                            help="Path to the file listing the images in the dataset.")
        parser.add_argument("--ignore-label", type=int, default=self.IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--num-classes", type=int, default=self.NUM_CLASSES,
                            help="Number of classes to predict (including background).")
        # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
        #                     help="Where restore model parameters from.")
        parser.add_argument("--pretrained-model", type=str, default=self.PRETRAINED_MODEL,
                            help="Where restore model parameters from.")
        parser.add_argument("--save-dir", type=str, default=self.SAVE_DIRECTORY,
                            help="Directory to store results")
        parser.add_argument("--gpu", type=int, default=0,
                            help="choose gpu device.")
        return parser.parse_args()