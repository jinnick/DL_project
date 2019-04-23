#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pickle
import json

import matplotlib
import matplotlib.pyplot as plt



import datasets.dummy_datasets as dummy_datasets
import utils.vis as vis_utils
import class_filter
from class_filter import iter_through_cls as filter
import build_dict
from build_dict import object_dict
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--input_query', 
        nargs='+', 
        dest='query', 
        type=str, 
        help='input query of list objects')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--use-vg3k',
        dest='use_vg3k',
        help='use Visual Genome 3k classes (instead of COCO 80 classes)',
        action='store_true'
    )
    parser.add_argument(
        '--thresh',
        default=0.7,
        type=float,
        help='score threshold for predictions',
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_list_from_file(file_name):
    f = open(file_name, 'rb')
    all_seg = pickle.load(f)
    f.close()
    return all_seg
    

def main(args):
    dummy_coco_dataset = (
        dummy_datasets.get_vg3k_dataset()
        if args.use_vg3k else dummy_datasets.get_coco_dataset())

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]
    print(im_list)
    
    all_cls_boxes = read_list_from_file('./all_boxes_p2.pickle')
    all_cls_segms = read_list_from_file('./all_segms_p2.pickle')
    all_cls_keyps = read_list_from_file('./all_keys_p2.pickle')
    

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        im = cv2.imread(im_name)
        print(im_name)
        print(i)
        
        cls_boxes = all_cls_boxes[i]
        cls_segms = all_cls_segms[i]
        cls_keyps = all_cls_keyps[i]
        
        query_list = args.query
        indx = []
        for i in range(len(query_list)):
            indx.append(object_dict[query_list[i]])
        
        #indx = [1,3,6,7]
        cls_boxes_filtered,sign = filter(cls_boxes,indx)
        #print(cls_boxes_filtered)
        dpi=200
        if sign == 0:
           fig = plt.figure(frameon=False)
           fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
           ax = plt.Axes(fig, [0., 0., 1., 1.])
           ax.axis('off')
           fig.add_axes(ax)
           im[:, :, ::-1],  # BGR -> RGB for visualization
           ax.imshow(im) 
           output_name = os.path.basename(im_name) + '.pdf' 
           print(output_name)
           fig.savefig(os.path.join(args.output_dir, '{}'.format(output_name)), dpi=dpi)
           plt.close('all')
           
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes_filtered,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=2
        )
        
   
    
    #with open('all_boxes.pkl', 'w') as f:
        #pickle.dump(all_cls_boxes, f)
        
    #with open('all_segms.pkl', 'w') as f:
        #pickle.dump(all_cls_segms, f)
        
    #with open('all_keyps.pkl', 'w') as f:
       #pickle.dump(all_cls_keyps, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
