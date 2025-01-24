"""
Module to evaluate the AI

How to run from a terminal:
    1. activate your environment
    2. run: $ py evaluate.py <model or dataset>
        with optional arguments:    
            --material <NbSe2, Graphene, Mos2, BN, or WTe2> 
            --weights <NbSe2, Graphene, Mos2, BN, or WTe2>
            --weights_path <filename of NbSe2 weights>
            --dataset <val or test>
"""

import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Set the log level of tensorflow, see bep.utils for more information.

import tensorflow as tf

from tdmms.tdmcoco import (
    CocoConfig, 
    CocoDataset,
    evaluate_coco
)
from bep.utils import (
    load_train_val_datasets,
    load_train_val_datasets_tdmms,
    check_dir_setup,
    load_tdmms_weights
)
from bep.dataset import bepDataset

ROOT_DIR = os.path.abspath("../")
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib

category_mapping = {
    1: "Mono_",
    2: "Few_",
    3: "Thick_"
}

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

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'evaluation')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")

#--------------------------------------------------------------#
#                      EVALUATE DATASET                        #
#--------------------------------------------------------------#

def evaluate_dataset(material: str, dataset: str) -> None:
    """
    Function to evaluate the dataset. Returns the amount of images and class counts.
    
    Args:
        - material: NbSe2 (from BEP) or Graphene, Mos2, BN, WTe2 (from TDMMS)

    Data directory should be setup as the following:
    ROOT_DIR/
        DL_2DMaterials/
            Dataset_DL_2DMaterials/
                <material>/
                    train/
                    val/
                    test/
        data/
            annotations/ (.ndjson or .json)
                train.ndjson
                val.ndjson
            images/
                train/
                val/
                test/
    """
    if material == 'NbSe2':
        dataset_train, dataset_val, dataset_test = load_train_val_datasets(dataset, use_bs=True)
    else:
        dataset_train, dataset_val = load_train_val_datasets_tdmms(ROOT_DIR, material)
        dataset_test = None

    datasets = [('Train', dataset_train), ('Val', dataset_val), ('Test', dataset_test)]

    print('')
    for dataset in datasets:
        if dataset[1]:
            all_cls = [[category_mapping[j['category_id']]+material for j in i['annotations']] for i in dataset[1].image_info]
            all_cls = [i for j in all_cls for i in j]
            print('{}: {} images'.format(dataset[0], len(dataset[1].image_ids)))
            print('Class counts:')
            for cls in dataset[1].class_names[1:]:
                print('        {}: {} images, part {}'.format(cls, all_cls.count(cls), round(all_cls.count(cls)/len(all_cls),2)))
    
    return None

#--------------------------------------------------------------#
#                        EVALUATE MODEL                        #
#--------------------------------------------------------------#

class EvaluationConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    NUM_CLASSES = 1 + 3 + 0

def evaluate_model(material: str, weights: str, weights_path: str, dataset_type: str = 'val', use_bs: bool = False, data_dir: str = 'data_afm') -> None:
    """
    Function to evaluate a model on a dataset. The evaluation consists of calculating the
    precision and recall for multiple IoU thresholds.

    Args:
        - material: NbSe2 (from BEP) or Graphene, Mos2, BN, WTe2 (from TDMMS).
        - weights: Graphene, Mos2, BN, WTe2 (from TDMMS).
        - weights_path: specific filename af a weights file, used if weights
                        argument is set to NbSe2.
        - dataset: to use the validation or test data. Only applicable for the
                        NbSe2 data, not the TDMMS data

    Data directory should be setup as the following:
    ROOT_DIR/
        DL_2DMaterials/
            Dataset_DL_2DMaterials/
                <material>/
                    train/
                    val/
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
    config = EvaluationConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    print('Running evaluation using the {} data and {} weights..'.format(material, weights))

    if weights != 'NbSe2':
        weights_filename = load_tdmms_weights(weights)
        MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights_filename)
    else:
        MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights_path)

    model.load_weights(MODEL_PATH, by_name=True)

    if material == 'NbSe2':
        if use_bs:
            dataset_type = dataset_type + '_bs'
        print('Loading NbSe2 {} {} dataset'.format(dataset_type, data_dir))
        dataset = bepDataset()
        coco = dataset.load_dir(os.path.join(ROOT_DIR, data_dir), dataset_type, reload_annotations=True, return_coco=True)
        dataset.prepare()
    else:
        dataset = CocoDataset()
        val_type = dataset_type 
        coco = dataset.load_coco(
            os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', material),
            val_type,
            return_coco=True
        )
        dataset.prepare()

    print("Running evaluation on {} images.".format(len(dataset.image_ids)))
    evaluate_coco(model, dataset, coco, "bbox", material)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate model and dataset'
    )
    parser.add_argument(
        'command',
        help='model or dataset'
    )
    parser.add_argument(
        '--material', 
        required=False,
        default='NbSe2',
        help='NbSe2 or MoS2'
    )
    parser.add_argument(
        '--weights', 
        required=False,
        default='MoS2',
        help='NbSe2, Graphene, Mos2, BN, or WTe2'
    )
    parser.add_argument(
        '--weights_path', 
        required=False,
        default='nbse2_from_mos2_images_20_epochs_111.h5',
        help='File name of the weights file'
    )
    parser.add_argument(
        '--dataset', 
        required=False,
        default='data_afm',
        help='data_afm, data_simp_afm etc'
    )
    parser.add_argument(
        '--split', 
        required=False,
        default='val',
        help='val or test, only for NbSe2 '
    )

    args = parser.parse_args()

    check_dir_setup((0.8, 0.1, 0.1), args.dataset, use_bs=True)

    if args.command == 'dataset':
        evaluate_dataset(args.material, args.dataset)
    
    if args.command == 'model':
        evaluate_model(args.material, args.weights, args.weights_path, args.split, False, args.dataset)