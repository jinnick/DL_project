Natural Language Query Autocompletion for Segmentation
-----

Deep Learning Final Project 4995

Based largely on:
    
   - Learning to Segment Every Thing
       * Paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Learning_to_Segment_CVPR_2018_paper.pdf
       * github: https://github.com/ronghanghu/seg_every_thing
       




### Currently Implemented

Output a image with segmap/bbox based on input query object(s)

`
python2 infer_query_2.py \
    --input_query  glass  \
    --output-dir kid_bat/query_seg\
    --image-ext jpg \
    --thresh 0.5 --use-vg3k \
    kid_bat/input
`if no object is detected in the image, orginal input image will be output.



Description of code files:
* infer_query_2.py - infer the segmentation map and bounding box of input image based on input query object
* infer_simple.py - infer the segmentation mape and bounding boxes of all objects of input image.
* dummy_datasets.py - include all classes of objects in VG dataset.
* build_dict.py - helper function to build a dictionary for indexing all classes in VG3K classes of VG dataset.
* class_filter.py - helper function to set the bounding box of a object detected not present in input query object list to empy.
* vis.py - visualize the image with segmentation map and bounding box.

Dependencies:
* caffe2 - https://github.com/ronghanghu/seg_every_thing/blob/master/INSTALL.md
* detecteron - https://github.com/ronghanghu/seg_every_thing/blob/master/INSTALL.md
* numpy
* python 2.7
* OpenCV
* pycocotools - 


