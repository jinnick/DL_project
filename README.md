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
* build_dict.py - helper function to build a dictionary for indexing all classes in VG3K classes of VG dataset.
* filter_class.py - script for training a new langauge model
* dynamic.py - script for evaluating trained model on new users

New files for ReferIt:
* build_referit_data.py - preprocess referit queries to feed to LSTM
* build_coco_data.py - preprocess mscoco queries to feed to LSTM
