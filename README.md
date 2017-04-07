# lang-track
Tracking by Natural Language Specification

# How to use it

## Download and setup Caffe (our own branch)

1. Caffe branch [here](https://github.com/mathrho/lang-track/tree/langtrackV3) (Note: `langtrackV3` branch not `master` branch)
2. Compile Caffe with option 
```
WITH_PYTHON_LAYER = 1
```

## Download pre-trained models

1. Download natual language segmentation model [quva01](/home/zhenyang/Workspace/devel/project/vision/NLST/snapshots/lang_high_res_seg/_iter_25000.caffemodel)
and copy to `MAIN_PATH/snapshots/lang_high_res_seg/_iter_25000.caffemodel`

2. Download tracking model [quva01](/home/zhenyang/Workspace/devel/project/vision/NLST/VGG16.v2.caffemodel)
and copy to `MAIN_PATH/VGG16.v2.caffemodel`

## Run demo code

### ipython notebook code

1. Given an image and a natural language query, how to identify a target (used on the first query frame of a video)
```
demo/lang_seg_demo.ipynb
```

2. Given a visual target (a box identified from step 1) and a sequence of frames, how to track the object in all the frames
```
demo/lang_track_demo.ipynb.ipynb
```



