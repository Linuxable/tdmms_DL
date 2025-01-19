"""
Module to load datarows with pre-labels into Labelbox.com

Make sure that the root directory of the project contains a python file
called config.py that contains the Labelbox API key and project IDs.

How to run from a terminal:
    1. activating the specifc environment FOR Labebox
    2. run: $ py mal/upload.py <dataset> --project_id <labelbox project id>
        Example:
            $ py mal/upload.py batch4 --project_id xxxxxxxxx
"""

import os
import sys
import uuid
import json
import labelbox as lb
import argparse

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

from api_config import LABELBOX_API_KEY as API_KEY
import config

# API_KEY = api_config.LABELBOX_API_KEY
client = lb.Client(API_KEY)

data_pre_fix = 'test_'

def get_datarows(data: str, data_dir: str) -> list:
    data_rows = []
    data_path = os.path.join(ROOT_DIR, data_dir, 'images', data)
    images = os.listdir(data_path)
    for i in images:
        data_rows.append({
            "row_data": os.path.join(data_path, i),
            "global_key": data_pre_fix+i,
            "external_id": data_pre_fix+i,
        })

    return data_rows

# Create a dataset in Labelbox
# Each dataset is a batch from the data/ folder
def upload_dataset(data: str, data_rows: list):
    """
    Function to upload a dataset to Labelbox as dataset.
    The dataset name and global keys of the images must both
    be unique in the Labelbox workspace.

    Args:
        - data: which folder to upload to from ROOT_DIR/<data_dir>/images/
    """
    dataset_name = data_pre_fix + data

    print(f"Uploading '{data}' data under '{dataset_name}'..")

    dataset = client.create_dataset(name=dataset_name)
    task = dataset.create_data_rows(data_rows)
    task.wait_till_done()

    print(f"Failed data rows: {task.failed_data_rows}")
    print(f"Errors: {task.errors}")

    if task.errors:
        for error in task.errors:
            if 'Duplicate global key' in error['message'] and dataset.row_count == 0:
                print(f"Deleting empty dataset: {dataset}")
                dataset.delete()

    return data_rows

def link_dataset_to_project(data: str, project_id: str, data_rows: list):
    """
    Function to link a dataset to a labeling project of Labelbox.

    Args:
        - data: which dataset to link.
        - project_name: either 'test' or 'NbSe2'.
        - data_rows: which data_rows need to be linked.
    
    """
    if not project_id:
        project_id = config.LABELBOX_MAIN_PROJECT_ID

    project = client.get_project(project_id)
    dataset_name = data_pre_fix + data

    print(f"Linking '{data}' to '{project}' project..")

    global_keys = [i['global_key'] for i in data_rows]

    add_batch_to_project(project, dataset_name, global_keys)
    upload_annotations(project, global_keys, data)

def add_batch_to_project(project, batch: str, global_keys: list):
    """
    Function link a batch / dataset of Labelbox to a labeling project of Labelbox.

    Args:
        - project: which project the data should be linked to.
        - batch: which batch / dataset should be linked.
            Each batch in a project must have a unique name.
        - global_keys:  all the image filenames.
    
    """
    batch = project.create_batch(
        batch,
        global_keys=global_keys,
        priority=5
    )

    print(f"Batch: {batch}")

def link_annotations_to_datarows(global_keys: list, data: str) -> list:
    """
    Function that reads the created annotation .json files by 
    predict.py and links them to their corresponding global keys.
    
    Args:
        - global_keys: all image filenames that have a corresponding .json file.
    """
    print('Adding global keys to data rows..')
    label_ndjson = []

    data_dir = os.path.join(ROOT_DIR, 'mal', data)

    for global_key in global_keys:
        file_name = global_key.replace(data_pre_fix, '').split('.')[0] + '.json' 

        with open(os.path.join(data_dir, file_name)) as f:
            annotations = json.load(f)

        for annotation in annotations:
            annotation.update({
                "dataRow": {
                    "globalKey": global_key
                }
            })
            label_ndjson.append(annotation)

    return label_ndjson

def upload_annotations(project, global_keys: list, data: str) -> None:
    """
    Function to upload the annotations to Labelbox.

    Args:
        - project: to which project the annotations should be uploaded.
        - global_keys: all the image filenames.
    """

    print('Uploading annotations..')

    # Upload as pre-labels
    # upload_job = lb.MALPredictionImport.create_from_objects(
    #     client = client, 
    #     project_id = project.uid, 
    #     name="mal_job"+str(uuid.uuid4()), 
    #     predictions=link_annotations_to_datarows(global_keys, data)
    # )

    # Upload as ground truth
    upload_job = lb.LabelImport.create_from_objects(
        client = client, 
        project_id = project.uid, 
        name="label_import_job"+str(uuid.uuid4()),  
        labels=link_annotations_to_datarows(global_keys, data)
    )

    print(f"Errors: {upload_job.errors}")
    print(f"Status of uploads: {upload_job.statuses}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upload stored model predictions'
    )

    parser.add_argument(
        'command', 
        help='Which dataset to use.'
    )

    parser.add_argument(
        '--project_id', 
        required=False,
        default=None,
        help='The project ID to link the data to'
    )

    parser.add_argument(
        '--data', 
        required=False,
        default='data',
        help='Data directory'
    )

    args = parser.parse_args()

    data_rows = get_datarows(args.command, args.data)
    upload_dataset(args.command, data_rows)
    link_dataset_to_project(args.command, args.project_id, data_rows)
