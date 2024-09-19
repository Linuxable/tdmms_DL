# Mask R-CNN for 2D Materials Detection Fine-tuned on NbSe2

**This is a fork repository that is focussed on fine-tuning the original model on a custom dataset of NbSe2 flakes.**

This repository provides information for utilizing Deep-learning based image segmentation algorithms for detecting atomically thin two-dimensional materials in the optical microscopy images. The related publications are: 

* "Autonomous robotic searching and assembly of two-dimensional crystals to build van der Waals superlattices," Satoru Masubuchi *et al.*, Nature Communications **9**, Article number: 1413 (2018). https://www.nature.com/articles/s41467-018-03723-w

* "Deep-Learning-Based Image Segmentation Integrated with Optical Microscopy for Automatically Searching for Two-Dimensional Materials," Satoru Masubuchi *et al.*, npj 2D Materials and Applications **4**, 3 (2020). https://www.nature.com/articles/s41699-020-0137-z

We also provide the training datasets, and trained model weights at https://doi.org/10.6084/m9.figshare.11881053.

If this work helped your research, it would be greatly appreciated if you could cite the papers in your publications.

The codes are based on the implementation of Mask R-CNN by (https://github.com/matterport/Mask_RCNN) and tfserve by (https://github.com/iitzco/tfserve) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. 

## Remarks
I hope to continue developing 2DMMS to become a truly helpful tool for the research community of van der Waals heterostructures. Please feel free to email me with your feedback or any issues at: msatoru@iis.u-tokyo.ac.jp

