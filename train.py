"""
Module to train the AI

How to run from a terminal:
    1. activate your environment
    2. run: $ py train.py 
        with optional arguments:   
            --starting_material <MoS2, WTe2, Graphene or BN>
            --last_layers <True or False>
"""

import os
import sys
import argparse
import datetime
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf

from imgaug import augmenters as iaa
from tdmms.tdmcoco import CocoConfig
from bep.utils import (
    check_dir_setup, 
    load_train_val_datasets,
    load_tdmms_weights
)
from bep.dataset import bepDataset
from notifications.discord import notify

ROOT_DIR = os.path.abspath("../")
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from mrcnn import model as modellib

tf.random.set_seed(42)

BATCH_SIZE = 4

#--------------------------------------------------------------#
#                         SETUP GPU                            #
#--------------------------------------------------------------#

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#--------------------------------------------------------------#
#                       SETUP LOGS DIR                         #
#--------------------------------------------------------------#

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'training')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")

#--------------------------------------------------------------#
#                         TRAINING                            #
#--------------------------------------------------------------#

class TrainingConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE
    # LEARNING_MOMENTUM = 0.95
    NUM_CLASSES = 1 + 3 + 0

    def __init__(
        self,
        train_images: int,
        val_images: int,
        starting_material: str,
        last_layers: bool,
    ):
        super().__init__()
        batch_size = self.GPU_COUNT * self.IMAGES_PER_GPU
        total_image_count = train_images + val_images
        self.STEPS_PER_EPOCH = train_images / batch_size
        
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.CHECKPOINT_NAME = f'{date}_nbse2_simp_{starting_material.lower()}_{last_layers}_{total_image_count}_{batch_size}_'
        self.NAME = f'nbse2_simp_{starting_material.lower()}_{last_layers}_{total_image_count}_{batch_size}'

def train_model(
    starting_material: str = 'MoS2',
    last_layers: bool = False,
):
    """
    Function to train MRCNN.

    The epoch step size is the amount of iterations per epoch:
        step size = amount of images / batch size 
        batch size = gpu count * images per gpu

    MRCNN automatically saves the best weights during training.

    Args:
        - computer: if the data directories should be reloaded
        - starting_material: Which weights will be used for fine-tuning on NbSe2.
                    MoS2, BN, Graphene or WTe2
        - last_layers: True or False. Determines if the last layers weights are loaded or randomized.
            Requires a matching amount of classes between transfer and new model.
    
    Data directory should be setup as the following:
    ROOT_DIR/
        data/
            annotations/ (.ndjson or .json)
                train.ndjson
                val.ndjson
                test.ndjson
            images/
                train/
                val/
                test/
        weights/
            <material>_mask_rcnn_tdm_120.h5
    """

    # check_dir_setup((0.8, 0.1, 0.1), data='data_simp_afm', use_bs=True)
    dataset_train, dataset_val, _ = load_train_val_datasets('data_ex_afm', use_bs=True, train_val_only=True)

    config = TrainingConfig(
        len(dataset_train.image_ids),
        len(dataset_val.image_ids),
        starting_material,
        last_layers,
    )
    config.display()

    model = modellib.MaskRCNN(
        mode="training",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    MODEL_PATH = os.path.join(ROOT_DIR, 'weights', load_tdmms_weights(starting_material))
    print("Loading weights ", MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True, exclude=[] if last_layers == 'True' else ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    
    augmentation = get_augmentation()

    train_tdmms_stages(
        model,
        config,
        dataset_train,
        dataset_val,
        augmentation
    )

    return None

def get_augmentation():
    augmentation = iaa.SomeOf((0, None), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-180,180)),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.CropAndPad(percent=(-0.25, 0.25)),
        #iaa.Multiply((0.5, 1.5)),
        #iaa.GaussianBlur(sigma=(0.0, 0.5)),
        #iaa.AdditiveGaussianNoise(scale=(0, 0.15*255))
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.WithChannels(0, iaa.Multiply((0.5,1.5)))
        ),
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.WithChannels(1, iaa.Multiply((0.5,1.5)))
        ),
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.WithChannels(2, iaa.Multiply((0.5,1.5)))
        ),
        iaa.WithChannels(0, iaa.Multiply((0.5,1.5))),
        iaa.WithChannels(1, iaa.Multiply((0.5,1.5))),
        iaa.WithChannels(2, iaa.Multiply((0.5,1.5)))
    ])
    '''
    augmentation = iaa.SomeOf((0, 3), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)])
    ])
    '''

    return augmentation

def train_bep_stages(
    model: modellib.MaskRCNN,
    config: CocoConfig,
    dataset_train: bepDataset,
    dataset_val: bepDataset,
    augmentation
) -> None:
    notify('Started training {}'.format(config.CHECKPOINT_NAME[:-1]))

    notify('Training network heads')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=90,
        layers='heads',
        augmentation=augmentation
    )
    
    notify('Reduce LR and further tune all layers')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE/100,
        epochs=120,
        layers='all',
        augmentation=augmentation
    )

    notify('Done training')

    return None

def train_tdmms_stages(
    model: modellib.MaskRCNN,
    config: CocoConfig,
    dataset_train: bepDataset,
    dataset_val: bepDataset,
    augmentation
) -> None:
    notify('Started training {}'.format(config.CHECKPOINT_NAME[:-1]))

    # Training - Stage 1
    notify('Training network heads')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers='heads',
        augmentation=augmentation
    )

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    notify('Fine tune Resnet stage 4 and up')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE/10,
        epochs=60,
        layers='4+',
        augmentation=augmentation
    )
    
    
    # Training - Stage 3
    # Fine tune all layers
    notify('Fine tune all layers')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE /10,
        epochs=90,
        layers='all',
        augmentation=augmentation
    )
    
    notify('Reduce LR and further tune all layers')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE /100,
        epochs=120,
        layers='all',
        augmentation=augmentation
    )

    notify('Done training')

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model'
    )

    parser.add_argument(
        '--starting_material', 
        required=False,
        default='MoS2',
        help='MoS2, WTe2, Graphene or BN'
    )

    parser.add_argument(
        '--last_layers',
        required=False,
        default=False,
        help='True or False'
    )
    
    args = parser.parse_args()

    try:
        train_model(
            args.starting_material,
            args.last_layers,
        )
    except Exception as e:
        logging.error("An exception occurred", exc_info=True)
        notify("An error occurred during training:\n{}".format(e))