# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_class = 10
        self.num_class = 15 # CHANGED 
        self.num_heading_bin = 12
        self.num_size_cluster = 10
        self.num_size_cluster = 15 # CHANGED 

        self.type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
        self.type2class = {'bag':0, 'books':1, 'bookshelf':2,'shelves':2, 'box':3, 'cabinet':4, 'dresser':4, 'chair':5, 'desk':6, 'table':6, 'door':7, 'fridge':8, 'lamp':9, 'pillow':10, 'sink':11, 'sofa':12, 'tv':13, 'whiteboard':14} # CHANGED

        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
        self.type2onehotclass = {'bag':0, 'books':1, 'bookshelf':2,'shelves':2, 'box':3, 'cabinet':4, 'dresser':4, 'chair':5, 'desk':6, 'table':6, 'door':7, 'fridge':8, 'lamp':9, 'pillow':10, 'sink':11, 'sofa':12, 'tv':13, 'whiteboard':14} # CHANGED

        self.type_mean_size = {'bathtub': np.array([0.765840,1.398258,0.472728]),
                          'bed': np.array([2.114256,1.620300,0.927272]),
                          'bookshelf': np.array([0.404671,1.071108,1.688889]),
                          'chair': np.array([0.591958,0.552978,0.827272]),
                          'desk': np.array([0.695190,1.346299,0.736364]),
                          'dresser': np.array([0.528526,1.002642,1.172878]),
                          'night_stand': np.array([0.500618,0.632163,0.683424]),
                          'sofa': np.array([0.923508,1.867419,0.845495]),
                          'table': np.array([0.791118,1.279516,0.718182]),
                          'toilet': np.array([0.699104,0.454178,0.756250])}
        self.type_mean_size = {'bag': np.array([0.355188,0.378856,0.346023]),
                                'books': np.array([0.287604,0.295896,0.205587]),
                                'bookshelf': np.array([0.402985,1.063498,1.727273]),
                                'box': np.array([0.351994,0.364480,0.309092]),
                                'cabinet': np.array([0.571631,1.214407,0.963636]),
                                'chair': np.array([0.592329,0.552978,0.827272]),
                                'desk': np.array([0.688260,1.337694,0.737500]),
                                'door': np.array([0.160932,0.690090,1.880588]),
                                'dresser': np.array([0.528526,1.001342,1.183333]),
                                'fridge': np.array([0.732086,0.754600,1.650000]),
                                'lamp': np.array([0.367022,0.379614,0.690910]),
                                'pillow': np.array([0.355497,0.560770,0.318182]),
                                'sink': np.array([0.502248,0.599351,0.457344]),
                                'sofa': np.array([0.924369,1.875018,0.847046]),
                                'table': np.array([0.792666,1.285808,0.718182]),
                                'tv': np.array([0.248484,0.800022,0.608334]),
                                'whiteboard': np.array([0.140555,1.654753,1.045454])} # CHANGED
        

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb


