Natural Language Query Autocompletion for Segmentation
-----

Deep Learning Final Project 4995

Based largely on:
    
   - Learning to Segment Every Thing
       * Paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Learning_to_Segment_CVPR_2018_paper.pdf
       * github: https://github.com/ronghanghu/seg_every_thing
       




### Currently Implemented

Train a query model with ReferIt captions using 

`
python query_completion/code/trainer.py 
`
Set hyperparameters following the format in `default_params.json`.

User embeddings currently locked, set to 0, and not trained.

Description of code files:
* beam.py - helper code for doing beam search
* factorcell.py - implementation of the FactorCell recurrent layer
* model.py - defines the Tensorflow graph for the language model
* trainer.py - script for training a new langauge model
* dynamic.py - script for evaluating trained model on new users

New files for ReferIt:
* build_referit_data.py - preprocess referit queries to feed to LSTM
* build_coco_data.py - preprocess mscoco queries to feed to LSTM
