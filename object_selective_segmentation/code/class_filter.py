#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:55:50 2019

@author: zyy
"""
import numpy as np
#import np.array as array
def mask_seg_query(cls_boxes,indx):
    cls_boxes = np.asarray(cls_boxes)
    mask = np.zeros((cls_boxes.shape[0],cls_boxes.shape[1]),dtype=int)
    for i in indx:
        mask[i,:] = np.ones((1,5),dtype=int)
    cls_boxes =  np.multiply(cls_boxes, mask)
    cls_boxes = cls_boxes.tolist(cls_boxes)
    return cls_boxes
    
def iter_through_cls(cls_boxes,indx):
    cls_boxes1 = cls_boxes
   
    for i in range(len(cls_boxes)):
        sign = 0
        if cls_boxes[i] != []:
           
            for j in range(len(indx)):
                 
                if i == indx[j]:
                    sign = 1
                    break
            if sign == 0:
                cls_boxes1[i] = []
    #print(cls_boxes)       
    return cls_boxes1,sign