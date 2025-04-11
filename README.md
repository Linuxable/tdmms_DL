# Mask R-CNN for 2D Materials Detection Fine-tuned on NbSe2

This repository contains the model, training, and evaluation code for 2D materials detection using Mask R-CNN. The goal of this project is to fine-tune the model of [Masubuchi *et al.*](https://www.nature.com/articles/s41699-020-0137-z), which was created for graphene, MoS2, hBN and WTe2, on NbSe2. This repository is built on the repository of [Masubuchi *et al.*](https://github.com/tdmms/tdmms_DL).

>This is the original publication: "Deep-Learning-Based Image Segmentation Integrated with Optical Microscopy for Automatically Searching for Two-Dimensional Materials," Satoru Masubuchi *et al.*, npj 2D Materials and Applications **4**, 3 (2020). https://www.nature.com/articles/s41699-020-0137-z

The orignal datasets and model weights can be found at: https://doi.org/10.6084/m9.figshare.11881053.

This model uses `tensorflow-gpu 2.4.0` and `python 3.8.10`. For GPU compatibility see this [link](https://www.tensorflow.org/install/source#gpu).

## Setup

Activate your environment and run `pip install -r requirements.txt`. For Conda and GPU usage please scroll down.
The directory should be setup as follows:

- Root directory/
	- data/ *contains the data for this project*
		- annotations/
			- train.ndjson
			- val.ndjson
			- test.ndjson
			- batch1.ndjson
			- batch2.ndjson
			- batchN.ndjson
	- images/
		- train/
		- val/
		- test/
		- batch1/
		- batch2/
		- batchN/

- DL_2DMaterials/ *original tdmms dataset*
- logs/
- mal/ *used for temporary files for MAL*
- tddms_DL/ *this repository*
- weights/ *contains all model weights, also from tdmms*
- api_config.py *contains secret API keys for Labelbox and Pushover*

**Only the *batch* folders in the data directory and files are necessary.** The train, validation and test directories will be created automatically if the batch folders and files are present.  

## Usage
### Training

To train the model run `python train.py`, with optional arguments:
-  `--starting_material <MoS2, WTe2, Graphene or BN>`
- `--last_layers <True or False>`

### Evaluation
To evaluate the model or a dataset run `python evaluate.py <model or dataset>`, with optional arguments:
-  `--material <NbSe2, Graphene, Mos2, BN, or WTe2>` Default: `NbSe2`.
-  `--weights <MoS2 or NbSe2>` Default: `MoS2`.
-  `--weights_path` Default: `nbse2_from_mos2_images_20_epochs_111.h5`, only used if `--weights` set to `NbSe2`.
- `--dataset <data directory>` In this README called 'data'.
-  `--split <val or test>` Default: `test`, only used if `--weights` is set to `NbSe2`.

For manual interactive inspection the `inspect_model.ipynb` notebook can be used.  

During training weight checkpoints are saved to the `logs/training/` directory. To use these weights, copy them to the `weights/` directory.

### Model Assisted Labeling
For the use of model assisted labeling, please see [this document](https://github.com/Linuxable/tdmms_DL/blob/master/mal/README.md).

### Image Splitting Algorithm
For the use of image splitting, please see [this document](https://github.com/Linuxable/tdmms_DL/blob/master/image_splitting/README.md).

## Annotations

Images are labeled and annotated by [Labelbox](https://labelbox.com/). This results in `.ndjson` annotation files. These are automatically converted to COCO format `.json` files.

## How-To DelftBlue
[This document](https://github.com/Linuxable/tdmms_DL/blob/master/How-to-DelftBlue.md) explains how to setup and use the DelftBlue supercomputer of the TU Delft.

## Conda GPU Setup

To setup a Conda environment for GPU usage, run the following commands:

1.  `conda create -n <your env name> python=3.8.10 -c conda-forge`
2.  `conda activate <your env name>`
3.  `conda install cudatoolkit=11.0 -c conda-forge`
4.  `conda install cudnn=8.0 -c conda-forge`
5.  `cd <your root dir>/tdmms_DL`
6.  `pip install -r requirements.txt`

>In the How-To DelftBlue documentation this is done using a batchjob script.

And you are good to go!

## Contact

If you have any questions please send a mail to abelloekdelange@gmail.com.