import sys
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread, imsave
import numpy as np


# randomly sample images and gt_boxes and plot the boxes in image
def plot_annotations(roidb, size = 1):

    if not os.path.exists("./annotations"):
        os.mkdir("./annotations")

    # select images randomly
    rand_indices = np.random.randint(0, len(roidb), size= size)

    for ix in rand_indices:
        im = imread(roidb[ix]['image'])
        fig, ax = plt.subplots()
        ax.imshow(im)

        print('====> {}'.format(roidb[ix]['image']))
        for box in roidb[ix]['boxes']:
            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            print(box)

            # rect = patches.Rectangle((bottom_left_x, bottom_left_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        filename = roidb[ix]['image'].split("/")[-1].split(".")[0]


        fig.savefig('./annotations/{}_with_annotation.png'.format(filename))
        plt.close()

def check_rangegs(roidb):

        from pandas import DataFrame
        all_boxes = [(im['image'].split("/")[-1] , box[0], box[1], box[2], box[3]) for im in roidb for box in im['boxes']]
        df = DataFrame(all_boxes).set_index(0)
        df.columns = ['x1', 'y1', 'x2', 'y2']

        if ((df['x1'] > 2047).any() or (df['x2'] > 2047).any() or (df['y1'] > 1023).any() or (df['y1'] > 1023).any()):
            print("Bbox overflow!")

        print(df[df['x1']> 2047])