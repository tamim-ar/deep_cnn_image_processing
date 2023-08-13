# -*- coding: utf-8 -*-
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

import os
import random
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
from tqdm import tqdm
from .utils import LOG_INFO,create_dir
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#---------------------------------------------------------------
class Processor(object):
    def __init__(self,
                data_path,
                save_path,
                fmt,
                data_dim,
                image_type,
                data_size=1024,
                label_den='train'):
        '''
            initializes the class
            args:
                data_path   =   location of raw data folder which contains eval and train folder
                save_path   =   location to save outputs (tfrecords,config.json)
                fmt         =   format of the image
                data_dim    =   dimension to resize the images
                image_type  =   type of image (grayscale,rgb,binary)
                data_size   =   the size of tfrecords
                label_den   =   label denoter (by default : train)
        '''
        # public attributes
        self.data_path  =   data_path
        self.save_path  =   save_path
        self.fmt        =   fmt
        self.data_dim   =   data_dim
        self.image_type =   image_type
        self.data_size  =   data_size
        self.label_den  =   label_den
        # private attributes
        self.__train_path   =   os.path.join(self.data_path,'train')
        self.__eval_path    =   os.path.join(self.data_path,'eval')
        self.__test_path    =   os.path.join(self.data_path,'test')
        
        # output paths
        self.__tfrec_path   =   create_dir(self.save_path,'tfrecords')
        self.__tfrec_train  =   create_dir(self.__tfrec_path,'train')
        self.__tfrec_eval   =   create_dir(self.__tfrec_path,'eval')
        self.__tfrec_test   =   create_dir(self.__tfrec_path,'test')
        
        self.__config_json  =   os.path.join(self.save_path,'config.json')
        # map labels
        self.__labelMapper()
        # extract image paths
        self.__train_img_paths  =   self.__imgPathExtractor('train')
        self.__eval_img_paths   =   self.__imgPathExtractor('eval')
        self.__test_img_paths   =   self.__imgPathExtractor('test')
        

    def __labelMapper(self):
        '''
            maps the labels from label denoter
        ''' 
        label_ext   =   os.path.join(self.data_path,self.label_den)
        _labels     =   os.listdir(label_ext)
        _labels     =   sorted(_labels,key=str.lower)
        self.__labels   =   _labels
    
    def __imgPathExtractor(self,mode):
        '''
            image path  extractor
        '''
        _img_paths=[img_path for img_path in tqdm(glob(os.path.join(self.data_path,mode,f"*/*.{self.fmt}")))]
        random.shuffle(_img_paths)
        return _img_paths    


    
    def __getLabel(self,img_path):
        '''
            get label from data path
        '''
        _base  =  os.path.dirname(img_path)
        _label =  os.path.basename(_base)
        return self.__labels.index(_label)    

    def __toTfrecord(self):
        '''
        Creates tfrecords from Provided Image Paths
        '''
        tfrecord_name=f'{self.__rnum}.tfrecord'
        tfrecord_path=os.path.join(self.__rec_path,tfrecord_name) 
        LOG_INFO(tfrecord_path)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:    
            
            for img_path in tqdm(self.__paths):
                
                # image ops
                # read
                if self.image_type=='rgb':
                    img=cv2.imread(img_path)
                else:
                    img=cv2.imread(img_path,0)
                # resize
                img=cv2.resize(img,(self.data_dim,self.data_dim))

                if self.image_type=='binary':
                    # Otsu's thresholding after Gaussian filtering
                    blur = cv2.GaussianBlur(img,(5,5),0)
                    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
                # Png encoded data
                _,img_coded = cv2.imencode('.png',img)
                # Byte conversion
                image_png_bytes = img_coded.tobytes()

                # label 
                label=self.__getLabel(img_path)
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes),
                        'label':_int64_feature(label)
                }
                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            
    def __create_df(self):
        '''
            tf record wrapper
        '''
        for idx in range(0,len(self.__img_paths),self.data_size):
            self.__paths      =   self.__img_paths[idx:idx+self.data_size]
            self.__rnum       =   idx//self.data_size
            self.__toTfrecord()

    def process(self):
        '''
            routine to create output
        '''
        # create tf recs
        ## train
        self.__img_paths=self.__train_img_paths
        self.__rec_path =self.__tfrec_train
        self.__create_df()
        ## eval 
        self.__img_paths=self.__eval_img_paths
        self.__rec_path =self.__tfrec_eval
        self.__create_df()
        ## test 
        self.__img_paths=self.__test_img_paths
        self.__rec_path =self.__tfrec_test
        self.__create_df()
        
        # config.json
        if self.image_type=='rgb':
            _channels=3
        else:
            _channels=1
        
        _config={'img_dim':self.data_dim,
                'nb_channels':_channels,
                'image_type':self.image_type,
                'nb_classes':len(self.__labels),
                'nb_train_data':len(self.__train_img_paths),
                'nb_eval_data':len(self.__eval_img_paths),
                'labels':self.__labels
                }
        with open(self.__config_json, 'w') as fp:
            json.dump(_config, fp,sort_keys=True, indent=4)
        
