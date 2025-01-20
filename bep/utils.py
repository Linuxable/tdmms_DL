"""Module that contains used functions"""

import json
import os
import random
import shutil
import matplotlib.pyplot as plt
import skimage


# Code for debugging -------------------------------------------
# import sys
# ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
# print('Root directory:',ROOT_DIR)
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
# ----------------------------------------------------------------

from bep.dataset import bepDataset
from bep.exceptions import WeightsNotFound, IncorrectDataSplit
from tdmms.tdmcoco import CocoDataset, CocoConfig

from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.model import log

from typing import Tuple, Union, List

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))

LABELBOX_NBSE2_SIO2_ID = 'cm167pqz802tq07023jfr2abh'
LABELBOX_NBSE2_SIO2_EXTENDED_ID = 'cm46uxql6023s07yx92im8tt4'


_data_sets = ['train', 'val', 'test']
data_types = ['images', 'annotations']

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                               TENSORFLOW LOGGING LEVELS                                   #
#                                                                                           #
#-------------------------------------------------------------------------------------------#

""""
Tensorflow logging levels:

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed

Set by running: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                              FUNCTIONS TO LOAD DATASETS                                   #
#                                                                                           #
#-------------------------------------------------------------------------------------------#

def load_train_val_datasets(data: str = 'data', use_bs: bool = False) -> Tuple[bepDataset, bepDataset]:
    """
    Function to load train and validation datasets of the BEP data.
    
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
    """
    data_sets = _data_sets
    if use_bs:
        data_sets = [i+'_bs' for i in _data_sets]

    dataset_train = bepDataset()
    dataset_train.load_dir(os.path.join(ROOT_DIR, data), data_sets[0], reload_annotations=True)
    dataset_train.prepare()

    dataset_val = bepDataset()
    dataset_val.load_dir(os.path.join(ROOT_DIR, data), data_sets[1], reload_annotations=True)
    dataset_val.prepare()

    dataset_test = bepDataset()
    dataset_test.load_dir(os.path.join(ROOT_DIR, data), data_sets[2], reload_annotations=True)
    dataset_test.prepare()

    return dataset_train, dataset_val, dataset_test

def load_train_val_datasets_tdmms(material: str = 'MoS2') -> Tuple[CocoDataset, CocoDataset]:
    """
    Function to load train and validation datasets of the TDMMS data.

    Materials: BN, Graphene, MoS2, WTe2

    Data directory should be setup as the following:
    ROOT_DIR/
        DL_2DMaterials/
            Dataset_DL_2DMaterials/
                <material>/
                    train/
                    val/

    Args:
        - material: material to load the data from
    """
    dataset_train = CocoDataset()
    dataset_train.load_coco(os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', material), 'train')
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_coco(os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', material), 'val')
    dataset_val.prepare()

    return dataset_train, dataset_val

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                                 FUNCTION TO LOAD TDMMS WEIGHTS                            #
#                                                                                           #
#-------------------------------------------------------------------------------------------#

def load_tdmms_weights(material: str) -> str:
    if material.lower() == 'mos2':
        return 'mos2_mask_rcnn_tdm_0120.h5'
    elif material.lower() == 'bn' or material.lower() == 'hbn':
        return 'bn_mask_rcnn_tdm_120.h5'
    elif material.lower() == 'graphene':
        return 'graphene_mask_rcnn_tdm_0120.h5'
    elif material.lower() == 'wte2':
        return 'wte2_mask_rcnn_tdm_0071.h5'
    else:
        raise WeightsNotFound(f'Weights for {material} not found.')


#-------------------------------------------------------------------------------------------#
#                                                                                           #
#              FUNCTIONS TO SPLIT BATCHES INTO TRAIN, VALIDATION AND TEST SETS              #
#                                                                                           #
#-------------------------------------------------------------------------------------------#


def check_dir_setup(
    data_split: Tuple[float,float,float],
    data: str = 'data',
    use_bs: bool = False,
) -> None:
    """Function to check if the directory is setup correctly. This 
    means checking for train and validation folders/files.

    If there are no train and or validation folders/files then these will
    be created from the found batch folders/files.

    Like:
        data/
            annotations/ (.ndjson or .json)
                batch1.ndjson
                batch2.ndjson
                .
                .
                train.ndjson
                val.ndjson
                test.ndjson
            images/
                batch1/
                    [image_1_name].png
                    [image_2_name].png
                    .
                    .
                batch2/
                .
                .
                train/
                val/
                test/

    Args:
        - data_split: determines the train validation test split
                    (train size, validation size, test size)
                    For example: (0.8, 0.1, 0.1)
    """
    data_sets = _data_sets
    if use_bs:
        data_sets = [i+'_bs' for i in _data_sets]

    for dt in data_types:
        for ds in data_sets:
            extension = ''
            if dt == data_types[1]:
                extension = '.ndjson'
            
            path = os.path.join(ROOT_DIR, data, dt, ds+extension)
            if not os.path.exists(path):
                print(f'{path} did not exist')
                create_dir_setup(ROOT_DIR, data, data_split)
                return None
            
    print('Directory setup correctly')

    return None

def create_dir_setup(
    data_split: Tuple[float, float, float],
    data: str = 'data',
    use_bs: bool = False,
) -> None:
    """Function to reset and create train and validation directories."""
    
    print('Creating directories from batches')

    batches = [i.split('.')[0] for i in os.listdir(os.path.join(ROOT_DIR, data, 'annotations'))]

    data_sets = _data_sets
    if use_bs:
        data_sets = [i+'_bs' for i in data_sets]

    batches = [
        i 
        for i in os.listdir(os.path.join(ROOT_DIR, data, 'images'))
        if ('batch' in  i and (i != 'batchsplit' or use_bs))
    ]

    print('Found batches:',', '.join(batches))

    images_for_training = []
    split_images = []
    if not use_bs:
        images_for_training = get_images_for_training(batches, data, 15)
    else:
        if os.path.isdir(os.path.join(ROOT_DIR, data, 'images', 'batchsplit')):
            split_images = list(set([i.split('_split')[0] for i in os.listdir(os.path.join(ROOT_DIR, data, 'images', 'batchsplit'))]))
    
    reset_dirs(data_sets, data)
    data_split_images(batches, data_split, images_for_training, data_sets, split_images, data)
    data_split_annotations(batches, data_sets, data)

    return None
    
def reset_dirs(data_sets: list, data: str) -> None:
    """Function to reset the image and annotation directories of the
    train and validation sets."""
    # Reset image directory
    for ds in data_sets:
        path = os.path.join(ROOT_DIR, data, 'images', ds)
        
        if os.path.exists(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                print('Error in deleting image directory:',e)
        
        os.mkdir(os.path.join(ROOT_DIR, data, 'images', ds))
                
    # Reset annotations directory
    for ds in data_sets:
        path = os.path.join(ROOT_DIR, data, 'annotations', ds+'.ndjson')
        
        if os.path.exists(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                print('Error in deleting annotations file:',e)
                
    return None

def get_images_for_training(batches: list, data: str, annotation_threshold: int = 15):
    images = []
    for batch in batches:
        rows = []

        try:
            with open(os.path.join(ROOT_DIR, data, 'annotations', batch+'.ndjson')) as f:
                rows += [json.loads(l) for l in f.readlines()]
        except FileNotFoundError:
            print('{}.ndjson not found. Skipping'.format(batch))

        
        for row in rows:
            annotation_count = sum(
                len(label['annotations']['objects'])
                for label in list(row['projects'].values())[0]['labels']
            )

            if annotation_count >= annotation_threshold:
                images.append((batch, row['data_row']['external_id']))

    print('Images forced for training directory: {}'.format(images))
    return images

def data_split_images(
    batches: list,
    data_split: Tuple[float, float, float],
    images_for_training: List[str],
    data_sets: List[str],
    split_images: List[str],
    data: str
) -> None:
    """
    Function to load the images from all found batches and split the 
    images into a train and validation set.
    
    Args:
        - batches: list of all the found batch folders
        - train_size: size of the train split, also determines the val split
    """
    img_dir = os.path.join(ROOT_DIR, data, 'images')
    ann_dir = os.path.join(ROOT_DIR, data, 'annotations')

    imgs_batches = []
    
    for batch in batches:
        if (
            os.path.exists(os.path.join(ann_dir, batch+'.ndjson')) or
            os.path.exists(os.path.join(ann_dir, batch+'.json'))
        ):
            
            imgs_batches.append((batch, os.listdir(os.path.join(img_dir, batch))))  
    imgs_batches = [[(i[0], j) for j in i[1]] for i in imgs_batches]
    imgs_batches = [i for j in imgs_batches for i in j]
    imgs_batches = [i for i in imgs_batches if i[1].split('.')[0] not in split_images]
    
    random.shuffle(imgs_batches)

    if images_for_training:
        for i in images_for_training:
            imgs_batches.insert(0, imgs_batches.pop(imgs_batches.index(i)))
    
    img_count = len(imgs_batches)    
    print(f'Total image count: {img_count}')
    
    train_amount = round(data_split[0]*img_count)
    val_amount = round(data_split[1]*img_count)
    test_amount = img_count - train_amount - val_amount

    slicing = [
        (data_sets[0], (None, train_amount)),
        (data_sets[1], (train_amount,train_amount+val_amount)),
        (data_sets[2], (val_amount+train_amount, None))
    ]
    
    print('Copying images')
    for sl in slicing:
        for i in imgs_batches[sl[1][0]:sl[1][1]]: # i = (batch_name, image_name)
            shutil.copy(
                os.path.join(img_dir, i[0], i[1]),
                os.path.join(img_dir, sl[0])
            )

    print('Checking image counts')
    check_count_list = [(train_amount, data_sets[0]), (val_amount, data_sets[1]), (test_amount, data_sets[2])]
    for i,j in check_count_list:
        folder_img_count = len([i for i in os.listdir(os.path.join(img_dir, j))])
        if i != folder_img_count:
            raise IncorrectDataSplit(f'Calculated amount of {j} images, {i}, is not equal to the acutal amount of images, {folder_img_count}, in the destinated folder')

    return None

def data_split_annotations(batches: list, data_sets: list, data: str) -> None:
    """Function to load the annotations from all found batches and split the 
    annotations into a train and validation set. Uses the already created 
    train/ val/ and test/ folders in images/ to know which annotations belongs
    where.
    
    TODO: 'f.write(str(row)+'\n')' writes the .ndjson lines as a string, containing
    ' instead of ", and ' is no .json. It work with a .replace when reading the files.
    But it can better be fixed here.
    """
    img_dir = os.path.join(ROOT_DIR, data, 'images')
    ann_dir = os.path.join(ROOT_DIR, data, 'annotations')

    rows = []
    for batch in batches:
        if (
            (os.path.exists(os.path.join(ann_dir, batch+'.json')) and
            not os.path.exists(os.path.join(ann_dir, batch+'.ndjson')))
            or batch == 'batchsplit'
        ):
            convert_coco_to_ndjson(
                os.path.join(ann_dir, batch+'.json'),
                os.path.join(ann_dir, batch+'.ndjson')
            )

        if (
            os.path.exists(os.path.join(ann_dir, batch+'.ndjson'))
        ):
            with open(os.path.join(ann_dir, batch+'.ndjson')) as f:
                rows += [json.loads(l) for l in f.readlines()]


    print('Creating and writing annotation files')
    for ds in data_sets:
        imgs = os.listdir(os.path.join(img_dir, ds))

        with open(os.path.join(ann_dir, ds+'.ndjson'), "w+") as f:
            for row in rows:
                if row['data_row']['external_id'] in imgs:
                    f.write(str(row)+'\n')

    return None

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                                      COCO TO LABELBOX                                     #
#                                                                                           #
#-------------------------------------------------------------------------------------------#


def convert_coco_to_ndjson(coco_file_path: str, ndjson_file_path: str) -> None:
    """
    Convert a COCO format JSON file to NDJSON format.
    
    Parameters:
        coco_file_path (str): Path to the input COCO JSON file.
        ndjson_file_path (str): Path to the output NDJSON file.
    """

    class_variable_mapping = {
        'mono': 1,  # 1 layer           / 0.7 nm
        'few': 2,   # 2 - 10 layers     / 1.4 - 7 nm 
        'thick': 3  # 10 - 40 layers    / 7 - 28 nm
    }

    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {image['id']: image for image in coco_data['images']}
    
    grouped_annotations = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in grouped_annotations:
            grouped_annotations[image_id] = []
        grouped_annotations[image_id].append(annotation)
    
    PROJECT_ID = LABELBOX_NBSE2_SIO2_ID
    if 'ex' in ndjson_file_path:
        PROJECT_ID = LABELBOX_NBSE2_SIO2_EXTENDED_ID
        class_variable_mapping['massive'] = 4
    
    class_variable_mapping_inverse = {v: k for k, v in class_variable_mapping.items()}

    with open(ndjson_file_path, 'w+') as ndjson_file:
        for image_id, image_data in images.items():
            data_row = {
                "id": image_data['id'],
                "external_id": image_data['file_name'],
            }
            
            media_attributes = {
                "height": image_data['height'],
                "width": image_data['width'],
                "asset_type": "image",
                "mime_type": "image/png",
                "exif_rotation": "1"
            }
            
            project_data = {
                PROJECT_ID: {
                    "labels": [{
                        'annotations': {
                            'classifications': [],
                            'objects': [
                                {
                                    "annotation_kind": "ImagePolygon",
                                    "name": class_variable_mapping_inverse[annotation['category_id']].title(),
                                    "id": annotation['id'],
                                    "polygon": [
                                        {"x": annotation['segmentation'][0][i], "y": annotation['segmentation'][0][i + 1]}
                                        for i in range(0, len(annotation['segmentation'][0]), 2)
                                    ],
                                    "value": class_variable_mapping_inverse[annotation['category_id']].lower()
                                }
                                for annotation in grouped_annotations[image_data['file_name']]
                            ]
                        }
                    }]
                }
            }
            
            ndjson_record = {
                "data_row": data_row,
                "media_attributes": media_attributes,
                "projects": project_data
            }
            
            ndjson_file.write(json.dumps(ndjson_record) + '\n')

    return None

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                           CLASS TO RUN THE MODEL FOR INSPECTION                           #
#                                                                                           #
#-------------------------------------------------------------------------------------------#


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class runModel():
    def __init__(
            self,
            model: modellib.MaskRCNN,
            config: CocoConfig, 
            dataset: Union[bepDataset, CocoDataset] = None,
            plot_size: int = 8,
            iteration_index: int = None,
        ) -> None:
        """
        Instantiate the class like:
            run_model = runModel(<MRCNN object>, <CocoConfig object>)

        In bep_inspect_model.ipynb the ussage is demonstrated.
        """
        self.model = model
        self.config = config
        self.image_id = None
        self.dataset = dataset
        self.plot_size = plot_size
        
        self.iteration_index = 0
        if iteration_index:
            self.iteration_index = iteration_index
    
    def run(
            self, 
            dataset: Union[bepDataset, CocoDataset] = None, 
            rand: bool = False, 
            image_idx: int = 0
        ) -> None:
        """
        Function to run the model on an image.
        
        Args:
            - dataset: if a dataset here is provided it overwrites a previously loaded dataset
                        in the object.
            - rand: True or False, if a random image is chosen from the dataset or not
            - image_idx: choose the image index from the dataset to use

        """
        if dataset:
            self.dataset = dataset
        assert self.dataset

        self.image_id = self.dataset.image_ids[image_idx]
        if rand:
            self.image_id = random.choice(self.dataset.image_ids)
            
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(self.dataset, self.config, self.image_id)

        info = self.dataset.image_info[self.image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], self.image_id, self.dataset.image_reference(self.image_id)))

        results = self.model.detect([image], verbose=1)

        title = self.model.name + ' Predictions'

        ax = get_ax(1, 1, self.plot_size)
        r = results[0]
        visualize.display_instances(
            image, r['rois'],
            r['masks'],
            r['class_ids'], 
            self.dataset.class_names, r['scores'],
            ax=ax,
            title=title
        )
        return None
    
    def gt(
            self, 
            dataset: Union[bepDataset, CocoDataset] = None, 
            rand: bool = False, 
            image_idx: int = None,
            iterate: bool = False,
            show_bbox: bool = True,
            filename: str = None,
        ) -> None:
        """Function to show the ground truth of the image, on which run() made predictions."""
        if dataset:
            self.dataset = dataset
        assert self.dataset

        if rand:
            self.image_id = random.choice(self.dataset.image_ids)
        elif image_idx:
            self.image_id = self.dataset.image_ids[image_idx]
            print('Running on:\n{}'.format(dataset.image_info[image_idx]))
        elif filename:
            all_image_ids = [i['id'] for i in self.dataset.image_info]
            image_id = [i['id'] for i in self.dataset.image_info if filename in i['path']][0]
            self.image_id = all_image_ids.index(image_id)
        elif iterate:
            try:
                self.image_id = self.dataset.image_ids[self.iteration_index]
                print('Running on:\n{}'.format(dataset.image_info[self.iteration_index]))
                self.iteration_index += 1
            except IndexError:
                print('Iterated through all images')
                return None
        elif not self.image_id:
            if self.image_id != 0:
                print("Either run .run() first, set 'rand' to True, set 'image_idx', set 'filename', or set 'iterate' to True.")

        image = self.dataset.load_image(self.image_id)

        height, width, _ = image.shape
        
        if height > width:
            print('Image is in portrait mode, cannot visualise')
            return None

        mask, class_ids = self.dataset.load_mask(self.image_id)

        image, _, scale, padding, _ = utils.resize_image(
            image, 
            min_dim=self.config.IMAGE_MIN_DIM, 
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding)
        bbox = utils.extract_bboxes(mask)

        ax = get_ax(1, 1, self.plot_size)
        title = 'Ground Truth'
        visualize.display_instances(
            image,
            bbox,
            mask,
            class_ids,
            self.dataset.class_names,
            ax=ax, 
            title=title,
            show_bbox=show_bbox
        )
        return None

    def run_from_path(self, path: str, use_ex: bool = False) -> None:
        """Run detection on a single image."""
        image = self.load_image(path)
        image = self.resize_image(image)

        results = self.model.detect([image], verbose=1)

        title = self.model.name + ' Predictions'

        class_names = ['','Mono', 'Few','Thick']
        if use_ex:
            class_names.append('Massive')

        ax = get_ax(1, 1, self.plot_size)
        r = results[0]
        visualize.display_instances(
            image, r['rois'],
            r['masks'],
            r['class_ids'], 
            ['','Mono', 'Few','Thick'], r['scores'],
            ax=ax,
            title=title
        )
        return None

    def load_image(self, path: str):
        """Load a specified image and return a [H,W,3] Numpy array."""
        image = skimage.io.imread(path)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def resize_image(self, image):
        """Resize an image according to configurations."""
        image, _, _, _, _ = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE
        )

        return image
            
    def __str__(self) -> str:
        s = 'Model: {}'.format(str(self.model))
        s += '\nDataset: {}'.format(self.dataset)

        return s

if __name__ == '__main__':
    pass