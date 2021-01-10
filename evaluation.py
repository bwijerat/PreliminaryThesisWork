
######################################################
# Evaluation code
# By: Brian Wijeratne
# Course: EECS6323
#
# Used to evaluate heatmap performance against ground truth

import os
import re
import cv2
import csv
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from optparse import OptionParser
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_uint
import copy
import xml.etree.ElementTree as ET


def evaluate(path_g, path_t, model_data):

    img_g = cv2.imread(path_g, 0)
    img_t = cv2.imread(path_t, 0)

    # correct all file names in folder
    if (img_t is None):
        t1, t2 = os.path.split(path_t)
        input_files2 = sort_alphanumeric([f for f in os.listdir(t1) if os.path.isfile(os.path.join(t1, f))])
        for f_name1 in input_files2:
            t5 = f_name1.split('.')
            t6, t7 = t5[0].split('_')
            f_name2 = ('frame_%05d' % (int(t7) - 700)) + '.jpg'

            # f_name2 = ('frame_%05d' % (int(t9))) + '.jpg'

            src = os.path.join(t1,f_name1)
            dst = os.path.join(t1,f_name2)
            os.rename(src, dst)

        img_t = cv2.imread(path_t, 0)

    img_g = (img_g > 127).astype(np.int_)
    img_t = (img_t > 127).astype(np.int_)

    img_r_neg = img_g - img_t   # TP = 0, FP = -1, TN = 0, FN = 1
    img_r_pos = img_g + img_t   # TP = 2, FP = 1, TN = 0, FN = 1

    TP = np.float(np.count_nonzero((img_r_neg == 0) * (img_r_pos == 2)))
    FP = np.float(np.count_nonzero((img_r_neg == -1) * (img_r_pos == 1)))
    TN = np.float(np.count_nonzero((img_r_neg == 0) * (img_r_pos == 0)))
    FN = np.float(np.count_nonzero((img_r_neg == 1) * (img_r_pos == 1)))

    total = img_g.shape[0] * img_g.shape[1]
    # precision
    if ((TP == 0.0) and (FP == 0.0)):
        precision = 0
    else:
        precision = TP / (TP + FP)
    # recall
    if ((TP == 0.0) and (FN == 0.0)):
        recall = 0
    else:
        recall = TP / (TP + FN)
    accuracy = (TP + TN) / total

    # F1
    if((precision == 0.0) and (recall == 0.0)):
        F1 = 0
    else:
        F1 = 2*(precision*recall)/(precision+recall)

    CC = stats.pearsonr(img_g.flatten(), img_t.flatten())
    if np.isnan(CC[0]):
        CC = [0.0,1.0]

    EMD = stats.wasserstein_distance(img_g.flatten(),img_t.flatten())
    if np.isnan(EMD):
        EMD = 0.0

    model_data.append([TP,FP,TN,FN,precision,recall,accuracy,F1,CC[0],EMD])


def sort_alphanumeric(l):
	""" Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanum_key)


def main():
    parser = OptionParser()

    parser.add_option("-i", "--input1",
                      dest="input1",
                      help="Ground truth directory",
                      type="string",
                      action="store"
                      )
    parser.add_option("-j", "--input2",
                      dest="input2",
                      help="Test directory",
                      type="string",
                      action="store"
                      )
    (options, args) = parser.parse_args()

    # Ground truth directory
    path_ground = options.input1
    input_files1 = sort_alphanumeric([f for f in os.listdir(path_ground) if os.path.isfile(os.path.join(path_ground, f))])

    # Testing directory
    path_testing = options.input2
    input_models = sort_alphanumeric([dI for dI in os.listdir(path_testing) if os.path.isdir(os.path.join(path_testing, dI))])

    csv_path = os.path.join(path_testing, 'a-output.csv')
    df_index = ['TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'accuracy', 'F1', 'CC', 'EMD']
    df = pd.DataFrame(index=df_index)


    for model in input_models:

        #path_truth = os.path.join(path_testing, model, 'cubic', 'thresh')
        path_truth = os.path.join(path_testing, model)
        model_data = []

        print('Processing ' + model + '...')

        for file in input_files1:

            # XML
            path_g = os.path.join(path_ground, file)
            path_t = os.path.join(path_truth, file)

            evaluate(path_g, path_t, model_data)
            pass

        len = model_data.__len__()
        df[model] = list(np.around((np.array(model_data).sum(axis=0))/len,decimals=4))
        print(df)

    print('Done processing models.')
    # output csv
    df.to_csv(csv_path)

if __name__== "__main__":
	main()
