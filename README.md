# lang-track
Tracking by Natural Language Specification

# How to use it

## download and setup caffe (our own branch)

1. Caffe branch [here] (https://github.com/mathrho/lang-track/tree/langtrackV3) (Note: langtrackV3 branch not master branch)
2. Compile caffe with option 
```
WITH_PYTHON_LAYER = 1
```

## download pre-trained model

dowload natual language segmentation model [here] (/home/zhenyang/Workspace/devel/project/vision/NLST/snapshots/lang_high_res_seg/_iter_25000.caffemodel)
to MAIN_PATH/snapshots/lang_high_res_seg/_iter_25000.caffemodel

dowload tracking model [here] (/home/zhenyang/Workspace/devel/project/vision/NLST/VGG16.v2.caffemodel)
to MAIN_PATH/VGG16.v2.caffemodel

## run demo code

###ipython inotebook code

Given a image and a natural language query, how to indentify a target (used on the first query frame of a video)
```
demo/lang_seg_demo.ipynb
```

Given a box (viusal target) and a sequence of frames, how to track the object in all the frames
```
demo/lang_track_demo.ipynb.ipynb
```



