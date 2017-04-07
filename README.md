# lang-track
Tracking by Natural Language Specification

# How to use it

## download and setup caffe (our own branch)

Caffe branch: https://github.com/mathrho/lang-track/tree/langtrackV3 (Note: langtrackV3 branch not master)
Compile caffe with option WITH_PYTHON_LAYER = 1


## download pre-trained model

dowload natual language segmentation model /home/zhenyang/Workspace/devel/project/vision/NLST/snapshots/lang_high_res_seg/_iter_25000.caffemodel 
to MAIN_PATH/snapshots/lang_high_res_seg/_iter_25000.caffemodel

dowload tracking model (VGG16) /home/zhenyang/Workspace/devel/project/vision/NLST/VGG16.v2.caffemodel 
to MAIN_PATH/VGG16.v2.caffemodel

## run demo code

ipython inotebook code
demo/lang_seg_demo.ipynb: given a image and a natural language query, how to indentify a target (used on the first query frame of a video)
demo/lang_track_demo.ipynb.ipynb: given a box (viusal target) and a sequence of frames, how to track the object in all the frames




