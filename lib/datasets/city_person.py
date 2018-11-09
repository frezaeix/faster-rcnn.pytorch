from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import json
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class city_person(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'city_person_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'city_person', 'gtBboxCityPersons')
        self._img_path = os.path.join(cfg.DATA_DIR, 'city_person', 'leftImg8bit')

        self._classes = ('__background__',  # always index 0
                         'ignore', 'pedestrian', 'rider', 'sitting person' , 'person (other)', 'person group')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb


        # CityPerson specific config options
        self.config = {'cleanup': True}

        assert os.path.exists(self._img_path), \
            'Image path does not exist: {}'.format(self._img_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # (F) extract the city part of the image index to include in path

        city = self._get_city(index)

        image_path = os.path.join(self._img_path, self._image_set,
                                  city, index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt

        # (F) loading all file name in all folders

        image_set_folder = os.path.join(self._img_path, self._image_set)
        assert os.path.exists(image_set_folder), \
            'Path does not exist: {}'.format(image_set_folder)

        # (F) iterate through all images and drop their ext
        image_index = [image[:-4] for city in os.listdir(image_set_folder) for image in os.listdir(os.path.join(image_set_folder, city))]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'city_person')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_city_person_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _get_city(self, index):

        return index.split("_")[0]

    def _load_city_person_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        city = self._get_city(index)

        # the annotation file has "gtBboxCityPerson" at the end of file name instead of "leftImg8bit"
        index = '_'.join(index.split("_")[:-1]) + '_gtBboxCityPersons'

        filename = os.path.join(self._data_path, self._image_set, city, index + '.json')



        with open(filename) as data_file:
            data = json.load(data_file)
            objs = data['objects']

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x, y, w, h = obj['bbox']

            # (F) this dataset use top left corner point and width and height to define the bbox
            # we convert this representation to top left corner point and bottom right cornet point

            x1 = x
            y1 = y
            # (F) there are some gt bbox with boundaries outside of the frame,
            # we limit the boundaries to the image frame

            # TODO replace the hardcoded width and height
            x2 = min(x + w, 2048 - 1)
            y2 = min(y + h, 1024 - 1)

            cls = self._class_to_ind[str(obj['label'])]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}


    def _get_city_person_results_file_template(self):
        # city_person/results/det_<image_set>_<class_name>.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(cfg.DATA_DIR, 'city_person' ,'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_city_person_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} CityPerson results file'.format(cls))
            filename = self._get_city_person_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    # TODO should we also do this adjustment? I'd leave it as is
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._data_path,
            self._image_set,
            '{:s}', # city
            '{:s}.json')

        cachedir = os.path.join(cfg.DATA_DIR,'city_person', 'annotations_cache')
        aps = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_city_person_results_file_template().format(cls)
            # TODO the city_person_eval should be changed
            rec, prec, ap = city_person_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_city_person_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_city_person_results_file_template().format(cls)
                os.remove(filename)


if __name__ == '__main__':
    d = city_person('train')
    res = d.roidb
    from IPython import embed;

    embed()
