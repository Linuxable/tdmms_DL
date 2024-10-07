"""Module to train the AI"""

import os
import sys
from imgaug import augmenters as iaa

from tdmcoco import CocoConfig

from bep_data import bepDataset
from bep_utils import check_dir_setup

from mrcnn import model as modellib

class TrainingConfig(CocoConfig):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

def train_model(ROOT_DIR):
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'weights', 'graphene_mask_rcnn_tdm_0120.h5') # Graphene COCO+2D
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    config = TrainingConfig()
    # config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=DEFAULT_LOGS_DIR)


    # Load weights
    print("Loading weights ", COCO_MODEL_PATH)

    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    # model.load_weights(COCO_MODEL_PATH, by_name=True)
        
    check_dir_setup(ROOT_DIR, 0.7)
    
    dataset_train = bepDataset()
    dataset_train.load_dir(os.path.join(ROOT_DIR, 'data'), 'train')
    dataset_train.prepare()

    dataset_val = bepDataset()
    dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val')
    dataset_val.prepare()

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
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads',
                augmentation=augmentation)
    
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=60,
    #             layers='4+',
    #             augmentation=augmentation)
    
    # Training - Stage 3
    # Fine tune all layers
    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE /10,
    #             epochs=90,
    #             layers='all',
    #             augmentation=augmentation)
    
    # print("Reduce LR and further tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE /100,
    #             epochs=120,
    #             layers='all',
    #             augmentation=augmentation)    
    
if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("../../")
    sys.path.append(ROOT_DIR)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_model(ROOT_DIR)