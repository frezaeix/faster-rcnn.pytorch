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
from datasets.city_person_eval.coco import COCO
from datasets.city_person_eval.eval_MR_multisetup import COCOeval

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
        self._json_val_path = os.path.join(self._data_path, self._image_set, 'val_gt.json')

        # TODO What should we do with background class?!
        self._classes = ('__background__',  # always index 0
                         'ignore', 'pedestrian', 'rider', 'sitting person', 'person (other)', 'person group')

        # self.ignore_in_eval = (True, False, False, False, False, True)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # CityPerson specific config options
        self.config = {'cleanup': False, 'use_salt': False, 'height': 1024, 'width': 2048}

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
        # extract the city part of the image index to include in path

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

        # loading all file name in all folders **ordered alphabetically** by name of folders first and then file names

        image_set_folder = os.path.join(self._img_path, self._image_set)
        assert os.path.exists(image_set_folder), \
            'Path does not exist: {}'.format(image_set_folder)

        # iterate through all images and drop their ext
        image_index = [image[:-4] for city in sorted(os.listdir(image_set_folder)) for image in sorted(os.listdir(os.path.join(image_set_folder, city)))]
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

            width = self.config['width']
            height = self.config['height']

            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

            cls = self._class_to_ind[str(obj['label'])]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # ds_utils.validate_boxes(boxes, width, height)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    # TODO should be implemented
    def _print_detection_eval_metrics(self, coco_eval):
        pass

    def _do_detection_eval(self, json_det_file, output_dir):

        assert os.path.exists(self._json_val_path), \
            'json file for grand truth annotations of validation set does not exist in path: {}'\
                .format(self._json_val_path)

        ann_type = 'bbox'

        res_file = open(os.path.join(output_dir, 'eval_results.txt'), "w")
        for id_setup in range(3, 4):
            cocoGt = COCO(self._json_val_path)
            cocoDt = cocoGt.loadRes(json_det_file)
            imgIds = sorted(cocoGt.getImgIds())
            cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate(id_setup)
            cocoEval.accumulate()
            cocoEval.summarize(id_setup, res_file)

        res_file.close()

    def _city_person_results_one_class(self, boxes, category_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': im_ind + 1,  # in gt json file for eval, images are indexed starting by 1
                  'category_id': category_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_city_person_results_file(self, all_boxes, res_file):
        # [{"image_id": 12,
        #   "category_id": 2,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            # TODO do we have a background class in out data?
            if cls == '__background__' or cls != 'pedestrian':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             self.num_classes - 1))
            # since in json eval file we start with zero index for classes,
            # but here __background__ is zero.
            category_id = cls_ind - 1

            results.extend(self._city_person_results_one_class(all_boxes[cls_ind], category_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_' +
                                         self._image_set + '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_city_person_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = city_person('train')
    res = d.roidb
    from IPython import embed;

    embed()
