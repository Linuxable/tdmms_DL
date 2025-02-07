"""
Module to split images with many segmentations into multiple images

How to run from a terminal:
    run $ py image_splitting/split.py
        Optional arguments:
            --annotation_threshold <integer>
            --border <integer>
            --data <data or data_ex>
        Example:
            $ py image_splitting/split.py --annotation_threshold 15 --border 15
"""

import cv2
import os
import sys
import json
import contextlib
import argparse
import random

from shapely.geometry import box

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from bep.utils import load_train_val_datasets
import image_splitting.utils as utils

def split_images(
    annotation_threshold: int = 15,
    border: int = 15,
    dataset: str = 'data',
    log_iteration: bool = False,
) -> None:
    """
    The function that splits images into multiple images.

    After splitting is good to manually check the results.
    In inspect_model.ipynb use: run_model.gt(dataset=split, iterate=True, show_bbox=False)

    Args:
        - annotation_threshold: int = what the annotation threshold is. Determines
            which images will be split.
        - border: int = with how many pixels bboxs need to be extended for overlap search
            and split image creation.
        - log_iteration: bool = whether to write images with drawn bboxs during overlap iterations
    """

    extension = ''
    if log_iteration:
        extension = '_log_it'


    utils.reset_image_dir(os.path.join(ROOT_DIR, dataset, 'images', 'batchsplit'+extension))

    data, _, _ = load_train_val_datasets(dataset, False)

    images = utils.get_images_to_split_from_dataset(data.image_info, annotation_threshold)

    # tiled_background, width, height = utils.create_background(100, os.path.join(ROOT_DIR, dataset, 'images', 'batch4', '67_sio2_NbSe2_Exfoliation_C5-84_f4_img.png'))

    image_id_positions = utils.get_image_ids_positions(data.image_info, images)

    print(f'Images: {images}')

    annotations_file = os.path.join(ROOT_DIR, dataset, 'annotations', f'batchsplit{extension}.json')
    annotations_dict = {
        "info": {
            "year": 2024,
            "description": "",
            "version": 1.0,
            "date_created": "",
            "url": "tudelft.nl",
            "contributor": "Abel de Lange"
        }, 
        "categories": [
            {"id": 1, "name": "Mono_NbSe2"},
            {"id": 2, "name": "Few_NbSe2"},
            {"id": 3, "name": "Thick_NbSe2"}
        ],
        "annotations": [],
        "images": []
    }

    IMGS_MARKED_DIR = os.path.join(ROOT_DIR, dataset, 'backgrounds', 'marked_sisio2', '100x')
    IMGS_UNMARKED_DIR = os.path.join(ROOT_DIR, dataset, 'backgrounds', 'unmarked_sisio2', '100x')

    for image_id in image_id_positions:
        image_info = data.image_info[image_id]

        if not '02_michiel_sio2_NbSe2_f3' in image_info['path']:
            continue

        wafer_type = utils.determine_wafer_type(image_info['path'])

        if wafer_type == 'unmarked':
            continue

        print('\nSplitting: {}'.format(image_info['path']))
        print('Wafer type: {}'.format(wafer_type))

        image = cv2.imread(image_info['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        already_added_ids_image = []
        
        for annotation in image_info['annotations']:
            annotation_id = annotation['id'] # Called loop annotation

            if annotation_id in already_added_ids_image:
                print(f'Annotation {annotation_id} already in an image due to earlier overlap, skipping')
                continue
            already_added_ids_image.append(annotation_id)
            
            already_loaded_ids_flake = [annotation_id]

            bg_image_filename, bg_image_dir = utils.get_bg_image(IMGS_MARKED_DIR, IMGS_UNMARKED_DIR, wafer_type)
            bg_image = cv2.imread(os.path.join(bg_image_dir, bg_image_filename))
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            height, width, _ = bg_image.shape

            first_overlap_check = True
            last_overlap_count = 0

            print(f'\nAnalysing annotation with id {annotation_id}')

            bbox = tuple(annotation['bbox'])
            print(f'Inital bbox: {bbox}')
            x, y, w, h = (int(i) for i in bbox)

            filename = image_info['path'].split('\\')[-1].split('.')[0] + f'_split_{annotation_id}' + '.png'

            bbox_coords_list = [utils.bbox_to_coords(bbox)]

            while first_overlap_check or last_overlap_count > 0:
                if log_iteration:
                    flake_image = utils.cut_out_flake(bg_image, bbox, image, border)
                    filename_it = image_info['path'].split('\\')[-1].split('.')[0] + f'_split_{annotation_id}' + '_bbox_' + '_'.join([str(i) for i in list(bbox)])+ '.png'
                    coords = utils.bbox_to_coords(bbox)
                    coords = [int(i) for i in coords]
                    cv2.rectangle(flake_image, (coords[0], coords[1]), (coords[2], coords[3]), color=1, thickness=2)
                    utils.store_image(ROOT_DIR, filename_it, flake_image, dataset, extension)

                if first_overlap_check:
                    print('\nRunning first overlap check..')
                    cut_out_bbox = box(x-border, y-border, x+w+border, y+h+border)
                else:
                    print('Running another overlap check..')
                    x, y, w, h = [int(i) for i in updated_bbox]
                    cut_out_bbox = box(x-border, y-border, x+w+border, y+h+border)

                overlapping_annotations, overlapping_ids, split_overlapping_ids = utils.check_overlap_image(
                    already_added_ids_image,
                    already_loaded_ids_flake,
                    image_info,
                    cut_out_bbox
                )

                if len(overlapping_ids) > 0:
                    already_added_ids_image += overlapping_ids
                    already_loaded_ids_flake += overlapping_ids
                    bbox_coords, updated_annotations = utils.add_overlapping_annotations(
                        annotation_id,
                        overlapping_annotations,
                        filename,
                    )

                    print(f'Overlap bboxs: {bbox_coords}')

                    annotations_dict['annotations'] += updated_annotations
                    bbox_coords_list += bbox_coords

                    if log_iteration:
                        coords = utils.bbox_to_coords(bbox)
                        coords = [int(i) for i in coords]
                        cv2.rectangle(flake_image, (coords[0], coords[1]), (coords[2], coords[3]), color=5, thickness=2)
                
                if split_overlapping_ids:
                    annotations_dict, bbox_coords = utils.change_split_image(split_overlapping_ids, filename, annotations_dict)
                    bbox_coords_list += bbox_coords

                    print(f'Split overlap bboxs: {bbox_coords}')
         
                updated_bbox = utils.coords_to_bbox(utils.extract_bbox_coords(bbox_coords_list))
                bbox = updated_bbox

                last_overlap_count = len(overlapping_ids)
                first_overlap_check = False

            annotations_dict = utils.update_annotations_dict(
                ROOT_DIR,
                filename,
                annotations_dict,
                annotation_id,
                annotation,
                image_id,
                width,
                height,
                bbox,
                dataset
            )

            print(f'Final bbox: {bbox}')
            print(f'Creating and storing image {filename}')

            flake_image = utils.cut_out_flake(bg_image, bbox, image, border)
            utils.store_image(ROOT_DIR, filename, flake_image, dataset)

    print('\nDeleting images with zero annotations')
    annotations_dict = utils.delete_zero_annotation_images(ROOT_DIR, annotations_dict.copy(), dataset)

    print('Storing annotations file')
    with open(annotations_file, 'w+') as f:
        json.dump(annotations_dict, f)
    
    return None

class NullWriter:
    def write(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model'
    )

    parser.add_argument(
        '--annotation_threshold', 
        required=False,
        default=15,
        help='An integer'
    )

    parser.add_argument(
        '--border', 
        required=False,
        default=15,
        help='An integer'
    )

    parser.add_argument(
        '--data', 
        required=False,
        default='data_afm',
        help='data_afm or data_ex_afm or data_ex_afm_human'
    )
    args = parser.parse_args()

    with contextlib.redirect_stdout(NullWriter):
        split_images(
            int(args.annotation_threshold),
            int(args.border),
            args.data,
            log_iteration=True
        )
        
    # split_images(
    #     int(args.annotation_threshold),
    #     int(args.border),
    #     args.data,
    #     log_iteration=False
    # )
