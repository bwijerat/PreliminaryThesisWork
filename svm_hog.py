
######################################################
# Preliminary thesis research code
# By: Brian Wijeratne
# Course: Thesis Research
#
# This source file contains a collage of work testing out the practicality of using relatively simple theory-based computer vision techiques for identifying vehicles.
#
# PRIMARY ATTEMPT:
# 1. Apply color segmentation with Felzenszwalb's efficient graph based segmentation post processed to form superclusters via graph cut 
# 2. Apply AIM Saliency via MATLAB implementation for areas of strong salience
# 3. Apply Objectiveness measure via MATLAB implementation for areas of strong objectness
#
# Graph cut color map result is post processed with K-Means with automatic selection of optimal number of color clusters via eigen decomposition of an affinity matrix.
#
# SECONDARY ATTEMPT:
# Attempted to build a vehicle classifier via SVM trained from Histogram of Orientated Gradients of a small custom dataset.
#
# Salient image cropping was applied to help with feature extraction and classification by prioritizing the content of the image to be completely the vehicle.
# The classifier and descriptors were saved to the disk via pickle


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import cv2
import sys
import copy
import pickle
import random

from PIL import Image
from sklearn.utils import Bunch
from SVMHOG.featuresourcer import FeatureSourcer

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics, datasets
from skimage import img_as_float, img_as_uint
from skimage.future import graph
import skimage.color as color
import skimage.segmentation as seg

import matlab.engine as mat
import matlab

from os import listdir
from os.path import isfile, join

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from numpy import linalg as LA

def getAffinityMatrix(coordinates, k=7):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.

    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates))

    # for each row, sort the distances ascendingly and take the index of the
    # k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T

    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix

def eigenDecomposition(A, plot=True, topK=5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)

    # if plot:
    #     plt.title('Largest eigen values of input matrix')
    #     plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
    #     plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors



def single_img_check(img_, eng, clf, IS_FILE, IS_CROP):

    if IS_FILE is 0:
        img_r = get_img(img_, IS_FILE)
    else:
        img_r = get_img(img_, IS_FILE)

    if IS_CROP is 1:
        img_crop = get_crop(img_r, eng)
    else:
        img_crop = img_r

    feature_params = {
        'color_model': 'yuv',  # hls, hsv, yuv, ycrcb
        'bounding_box_height': img_crop.shape[0],  # img_height
        'bounding_box_width': img_crop.shape[1],  # img_width
        'number_of_orientations': 11,  # 6 - 12
        'cells_per_grid': 16,  # 8x8, 16x16
        'cells_per_block': 2,  # 1x1, 2x2
        'do_transform_sqrt': True
    }

    source = FeatureSourcer(feature_params, img_crop)
    desc = source.features(img_crop)

    y_pred = clf.predict(desc.reshape(1, -1))

    print('\nPredicted value: %d' % y_pred[0])

def sort_alphanumeric(l):

	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanum_key)

def connected_components(binary, DEL_SMALL_REGIONS):
    # getting contours regions
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # remove any small components
    xywh_temp = []
    ii = -1
    for cnt in contours:
        ii += 1
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        # cnt_len = cv2.arcLength(cnt, True)  # run something before if statement to successfully run if statement
        xywh_temp.append([x1, y1, w1, h1])
    ii = -1

    if DEL_SMALL_REGIONS:
        for xywh in xywh_temp:
            ii += 1
            x1, y1, w1, h1 = xywh
            if (w1 < 32) or (h1 < 32):
                del contours[ii]
                ii -= 1
                binary[y1:y1 + h1, x1:x1 + w1] = 0

    # getting mask with connectComponents
    ret, labels = cv2.connectedComponents(binary)

    return ret, labels.astype(np.uint8)

def objectness_saliency_detection(eng, img, FIG, MAP):

    boxes = None
    boxes = np.array(eng.runObjectness(matlab.double(img.tolist()), 100))

    # adjust for matlab indexing
    boxes[:, 0:4] = np.round(boxes[:, 0:4] - 1)

    x1 = boxes[:, 0].astype(int)
    y1 = boxes[:, 1].astype(int)
    x2 = boxes[:, 2].astype(int)
    y2 = boxes[:, 3].astype(int)

    img_r = copy.deepcopy(img)
    map = np.zeros(img_r.shape[0:2], dtype=int).astype(np.uint8)

    for i in np.arange(x1.__len__()):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        cv2.rectangle(img_r, (x1[i], y1[i]), (x2[i], y2[i]), (r, g, b), 1)
        map[y1[i]: y2[i], x1[i]: x2[i]] += 1

    if MAP is 'SQRT':
        map_norm = np.sqrt(cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)).astype(np.uint8)
    elif MAP is 'CUBIC':
        map_norm = cv2.normalize(np.power(img_as_float(cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)), 3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        map_norm = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    map_binary = cv2.threshold(map_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    slc = None
    slc = np.uint8(np.asarray(eng.AIM(matlab.double(img.tolist()))))

    slc_c = np.power(img_as_float(cv2.blur(cv2.medianBlur(slc, 7), (3, 3))), 3)
    slc_cubic = cv2.normalize(slc_c, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    slc_binary = cv2.threshold(slc_cubic, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    slc_map = np.zeros((slc_binary.shape[0], slc_binary.shape[1], 3), dtype=np.uint8)
    slc_map[:, :, 0] = slc_binary
    slc_map[:, :, 2] = map_binary

    slc_map_overlay = cv2.addWeighted(slc_map, 0.4, img, 1.0, 0)

    fig = plt.figure(FIG)
    fig.set_size_inches(18.5, 10.5)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(3, 2, 1); ax.imshow(img_r)
    ax = fig.add_subplot(3, 2, 2); ax.imshow(slc_map_overlay)
    ax = fig.add_subplot(3, 2, 3); ax.imshow(map)
    ax = fig.add_subplot(3, 2, 4); ax.imshow(map_binary, cmap='gray')
    ax = fig.add_subplot(3, 2, 5); ax.imshow(slc, cmap='gray')
    ax = fig.add_subplot(3, 2, 6); ax.imshow(slc_binary, cmap='gray')
    plt.close(fig)

    # plt.subplot(321), plt.imshow(img_r)
    # plt.subplot(322), plt.imshow(slc_map_overlay)
    # plt.subplot(323), plt.imshow(map)
    # plt.subplot(324), plt.imshow(map_binary, cmap='gray')
    # plt.subplot(325), plt.imshow(slc, cmap='gray')
    # plt.subplot(326), plt.imshow(slc_binary, cmap='gray')
    # plt.close(fig)

    return img_r, map, slc, map_binary, slc_binary, slc_map_overlay, fig


def run_object_detection(img_dir, img_out_dir, eng, clf, IS_OUTPUT):

    img_names = sort_alphanumeric([f for f in listdir(img_dir) if isfile(join(img_dir, f))])

    for img_ID, img_name in enumerate(img_names, start=0):

        if IS_OUTPUT is True:

            f_name, f_type = img_name.split('.')
            out_dir1 = os.path.join(img_out_dir, f_name)

            if not os.path.exists(out_dir1):
                os.mkdir(out_dir1)


        print("\nProcessing New Image.. Please wait.")

        img = cv2.imread(os.path.join(img_dir, img_name))

        _, _, _, map_binary, slc_binary, _, fig1 = objectness_saliency_detection(eng, img, FIG=1, MAP='SQRT')
        out_dir_1 = os.path.join(img_out_dir, 'fig1')
        if not os.path.exists(out_dir_1):
            os.mkdir(out_dir_1)
        fig1.savefig(os.path.join(out_dir_1, f_name + '_fig1.jpg'))


        ret_map, labels_map = connected_components(map_binary, DEL_SMALL_REGIONS=True)

        map_region_labels = np.unique(labels_map)[1:]

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img_c = seg.felzenszwalb(img_hsv)
        g = graph.rag_mean_color(img_hsv, img_c, mode='similarity')
        labels = graph.cut_normalized(img_c, g)
        out = color.label2rgb(labels, img_hsv, kind='avg')

        # Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        # ss.switchToSelectiveSearchFast()
        ss.switchToSelectiveSearchQuality()
        rects = ss.process()

        imOut = img.copy()
        ss_map = np.zeros(imOut.shape[0:2], dtype=int).astype(np.uint8)

        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < 100):
                x, y, w, h = rect

                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                cv2.rectangle(imOut, (x, y), (x + w, y + h), (r, g, b), 1, cv2.LINE_AA)
                ss_map[y:(y+h), x:(x+w)] += 1
            else:
                break



        # bounding boxes of found objects drawn on this img
        img_boxes = copy.deepcopy(img)

        img_list = []
        img_loc = []

        for id_N1, N1 in enumerate(map_region_labels, start=0):

            map_px = None
            map_px = np.where(labels_map == N1)

            x = np.min(map_px[1])
            y = np.min(map_px[0])
            w = np.max(map_px[1])
            h = np.max(map_px[0])

            img_crop = img[y:h, x:w, :]

            _, _, _, map_binary1, slc_binary1, _, fig2 = objectness_saliency_detection(eng, img_crop, FIG=2, MAP='CUBIC')

            out_dir_2 = os.path.join(img_out_dir, 'fig2')
            if not os.path.exists(out_dir_2):
                os.mkdir(out_dir_2)
            fig2.savefig(os.path.join(out_dir_2, f_name + '_fig2_' + str(id_N1) + '.jpg'))

            out1 = out[y:h, x:w, :]
            pts2 = np.stack((out1[:, :, 0].flatten(), out1[:, :, 1].flatten(), out1[:, :, 2].flatten()), axis=1)
            out1_col2 = np.unique(pts2, axis=0)

            affinity_matrix = getAffinityMatrix(out1_col2, k=7)
            k, _, _ = eigenDecomposition(affinity_matrix)
            k = np.sort(k)

            # k_num = k[int(np.ceil(k.__len__()/2))-1]
            k_num = max(k)

            kmeans = KMeans(n_clusters=k_num, init='k-means++', max_iter=k_num, n_init=10,
                            random_state=0)

            out1_col2_kmeans = kmeans.fit_predict(out1_col2)
            out1_col2_kmeans_u = np.unique(out1_col2_kmeans)

            out1_col3 = np.zeros(out1_col2.shape, dtype=np.uint8)

            for id_col2, col2 in enumerate(out1_col2_kmeans_u, start=0):

                idx = np.where(out1_col2_kmeans == col2)
                avg_col = np.average(out1_col2[idx], axis=0).astype(dtype=np.uint8)

                out1_col3[idx, :] = avg_col

            pts3 = np.zeros(pts2.shape, dtype=np.uint8)
            for id_col3, col3 in enumerate(out1_col2, start=0):

                pts3[np.where(pts2 == col3), :] = out1_col3[id_col3, :].astype(dtype=np.uint8)


            out2 = np.zeros(out1.shape, dtype=np.uint8)
            out2[:, :, 0] = pts3[:, 0].reshape(out2[:, :, 0].shape)
            out2[:, :, 1] = pts3[:, 1].reshape(out2[:, :, 0].shape)
            out2[:, :, 2] = pts3[:, 2].reshape(out2[:, :, 0].shape)


            for col in out1_col3:

                out1_col = (255 * (out2 == col)[:, :, 0]).astype(np.uint8)
                contours1 = cv2.findContours(out1_col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

                for id_cnt, cnt in enumerate(contours1, start=3):
                    # crop location
                    x1, y1, w1, h1 = cv2.boundingRect(cnt)
                    x2 = x1 + w1; y2 = y1 + h1
                    # absolute location
                    x1_ = x + x1; x2_ = x + x2; y1_ = y + y1; y2_ = y + y2

                    img_bin = img_crop[y1:y2, x1:x2]
                    cnt_sum = np.sum(slc_binary1[y1:y2, x1:x2]) / 255
                    cnt_cov = cnt_sum / ((y2 - y1 + 1) * (x2 - x1 + 1))

                    if cnt_cov > 0.75 or contours1.__len__() == 1:
                        # img_bin_overlay = cv2.addWeighted(cv2.cvtColor(slc_bin, cv2.COLOR_GRAY2RGB), 0.4, img_bin, 1.0, 0)
                        img_list.append(img_bin)
                        img_loc.append([x1, y1, x2, y2])

                        # single_img_check(img_bin, eng, clf, IS_FILE=1, IS_CROP=0)
                        # plt.figure(id_cnt)
                        # plt.imshow(img_bin)

                        r = random.randint(0, 255); g = random.randint(0, 255); b = random.randint(0, 255)

                        slc_binary1[y1:y2, x1:x2] = 0           # lowers chance of re-detecting salient region
                        cv2.rectangle(img_boxes, (x1_, y1_), (x2_, y2_), (r, g, b), 1)
                        cv2.putText(img_boxes, 'A', (x1_, y1_), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))


            id_cnt = id_cnt + 1
            fig3 = plt.figure(id_cnt)
            ax = fig3.add_subplot(1, 2, 1); ax.imshow(out1)
            ax = fig3.add_subplot(1, 2, 2); ax.imshow(out2)
            plt.close(fig3)

            out_dir_3 = os.path.join(img_out_dir, 'fig3')
            if not os.path.exists(out_dir_3):
                os.mkdir(out_dir_3)
            fig3.savefig(os.path.join(out_dir_3, f_name + '_fig3_' + str(id_N1) + '.jpg'))

            # plt.figure(id_cnt)
            # plt.subplot(221), plt.imshow(out1)
            # plt.subplot(222), plt.imshow(out1_gray)
            # plt.subplot(223), plt.imshow(out1_gray_thresh)

            plt.figure(0)
            plt.imshow(img_boxes)
            plt.close(fig='all')


        if IS_OUTPUT is True:

            out_dir_bb = os.path.join(img_out_dir, 'fig_bb')
            os.mkdir(out_dir_bb)
            cv2.imwrite(os.path.join(out_dir_bb, f_name + '_boundboxes.jpg'), img_boxes)

            for img_id, img_bin in enumerate(img_list, start=0):

                cv2.imwrite(os.path.join(out_dir1, str(img_id) + '.jpg'), img_bin)


def get_crop_ratio(img_r, eng):

    boxes = None
    boxes = np.array(eng.runObjectness(matlab.double(img_r.tolist()), 40))

    # adjust for matlab indexing
    boxes[:, 0:4] = np.round(boxes[:, 0:4] - 1)

    x1 = boxes[:, 0].astype(int)
    y1 = boxes[:, 1].astype(int)
    x2 = boxes[:, 2].astype(int)
    y2 = boxes[:, 3].astype(int)

    map = np.zeros(img_r.shape[0:2], dtype=int).astype(np.uint8)

    for i in range(boxes.shape[0]):
        map[y1[i]: y2[i], x1[i]: x2[i]] += 1

    slc = None
    slc = np.asarray(eng.AIM(matlab.double(img_r.tolist()))).astype(np.uint8)
    slc_sq = np.sqrt(slc).astype(np.uint8)

    img_map_n = cv2.normalize(map + slc_sq, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_map_n, kernel, iterations=2)

    img_binary = (255 * (img_dilation > 127)).astype(np.uint8)
    img_binary_stack = np.stack((img_binary, img_binary, img_binary), axis=2)
    img_dilation_overlay = cv2.addWeighted(img_binary_stack, 0.3, img_r, 1.0, 0)

    pixels = np.where(img_binary[:, :] == 255)
    max_y = np.max(pixels[0])
    min_y = np.min(pixels[0])
    max_x = np.max(pixels[1])
    min_x = np.min(pixels[1])

    y1_ratio = np.round(min_y / img_r.shape[0], 4)
    y2_ratio = np.round(max_y / img_r.shape[0], 4)

    x1_ratio = np.round(min_x / img_r.shape[1], 4)
    x2_ratio = np.round(max_x / img_r.shape[1], 4)

    return y1_ratio, y2_ratio, x1_ratio, x2_ratio

def get_crop(img_r, eng):

    boxes = np.array(eng.runObjectness(matlab.double(img_r.tolist()), 40))

    # adjust for matlab indexing
    boxes[:, 0:4] = np.round(boxes[:, 0:4] - 1)

    x1 = boxes[:, 0].astype(int)
    y1 = boxes[:, 1].astype(int)
    x2 = boxes[:, 2].astype(int)
    y2 = boxes[:, 3].astype(int)

    map = np.zeros(img_r.shape[0:2], dtype=int).astype(np.uint8)

    for i in range(boxes.shape[0]):
        map[y1[i]: y2[i], x1[i]: x2[i]] += 1

    slc = np.asarray(eng.AIM(matlab.double(img_r.tolist()))).astype(np.uint8)
    slc_sq = np.sqrt(slc).astype(np.uint8)

    img_map_n = cv2.normalize(map + slc_sq, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_map_n, kernel, iterations=2)

    img_binary = (255 * (img_dilation > 127)).astype(np.uint8)
    img_binary_stack = np.stack((img_binary, img_binary, img_binary), axis=2)
    img_dilation_overlay = cv2.addWeighted(img_binary_stack, 0.3, img_r, 1.0, 0)

    pixels = np.where(img_binary[:, :] == 255)
    max_y = np.max(pixels[0])
    min_y = np.min(pixels[0])
    max_x = np.max(pixels[1])
    min_x = np.min(pixels[1])

    # cv2.rectangle(img_r, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)

    img_c = img_r[min_y:max_y, min_x:max_x]
    base = 16; px = 256

    h = px; w = px

    ratio = np.double(img_c.shape[0]) / img_c.shape[1]
    if img_c.shape[0] > img_c.shape[1]:
        h = px
        w = int(px / ratio)
    elif img_c.shape[0] < img_c.shape[1]:
        w = px
        h = int(px * ratio)

    base_h = int(base * round(float(w) / base))
    base_w = int(base * round(float(h) / base))

    img_crop2 = cv2.resize(img_c, (base_h, base_w))

    # plt.figure(2)
    # plt.subplot(321), plt.imshow(img_r)
    # plt.subplot(322), plt.imshow(img_crop2)
    # plt.subplot(323), plt.imshow(img_map_n, cmap='gray')
    # plt.subplot(324), plt.imshow(img_dilation, cmap='gray')
    # plt.subplot(325), plt.imshow(img_binary, cmap='gray')
    # plt.subplot(326), plt.imshow(img_dilation_overlay, cmap='gray')

    # print(1)

    return img_crop2

def get_img(img_, IS_FILE):

    if IS_FILE is 0:
        img = cv2.imread(img_)
    else:
        img = img_

    px = 256

    h = px; w = px

    if (img.shape[0] > px) or (img.shape[1] > px):
        img_PIL = Image.fromarray(img)
        img_PIL.thumbnail((px, px), Image.ANTIALIAS)
        img_r = np.asarray(img_PIL)
        h = img_r.shape[0]
        w = img_r.shape[1]
    elif (img.shape[0] is not px) or (img.shape[1] is not px):
        img_r = img
        ratio = np.double(img.shape[0])/img.shape[1]
        if img.shape[0] > img.shape[1]:
            h = px
            w = int(px / ratio)
        elif img.shape[0] < img.shape[1]:
            w = px
            h = int(px * ratio)
    else:
        img_r = img

    base = 16
    base_h = int(base * round(float(w) / base))
    base_w = int(base * round(float(h) / base))

    img_r2 = cv2.resize(img_r, (base_h, base_w))

    return img_r2

def load_image_desc(path, labels, eng):
    desc_list = []
    target = []

    for id_sample, sample in labels.iterrows():

        if any(c.isalpha() for c in sample.id):
            img_path = os.path.join(path, sample.id + '.jpg')
            img_r_path = os.path.join(path, 'resize', sample.id + '.jpg')
            if sample.value is not 0:
                print('WRONG TARGET VALUE %d' % sample.value)
                break
        else:
            img_path = os.path.join(path, '%06d.jpg' % int(sample.id))
            img_r_path = os.path.join(path, 'resize', '%06d.jpg' % int(sample.id))
            if sample.value is not 1:
                print('WRONG TARGET VALUE %d' % sample.value)
                break

        img_r = get_img(img_path, img_r_path)
        img_crop, feature_params = get_crop(img_r, img_r_path, eng)

        source = FeatureSourcer(feature_params, img_crop)
        vehicle_features = source.features(img_crop)

        # rgb_img, y_img, u_img, v_img = source.visualize()
        # plt.figure(2)
        # plt.subplot(221), plt.imshow(rgb_img)
        # plt.subplot(222), plt.imshow(y_img, cmap='gray')
        # plt.subplot(223), plt.imshow(u_img, cmap='gray')
        # plt.subplot(224), plt.imshow(v_img, cmap='gray')

        print(str(id_sample) + '/' + str(labels.__len__()) + ': ' + str(vehicle_features.__len__()))

        desc_list.append(vehicle_features)
        target.append(sample.value)

    sys.stdout.write('\n')

    return Bunch(data=desc_list, target=target)


if __name__ == '__main__':
    # plt.interactive(True)

    eng = mat.start_matlab()
    eng.addpath(r'/home/bwijerat/Documents/Thesis/objectness-release-v2.2/objectness-release-v2.2', nargout=0)
    eng.addpath(r'/home/bwijerat/Documents/Thesis/ext-code/Project/AIM', nargout=0)
    eng.startup(nargout=0)

    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # img_path = '/home/bwijerat/Documents/Thesis/ext-code/Project/car-imgs/train_test/000178.jpg'
    # img_r_path = '/home/bwijerat/Documents/Thesis/ext-code/Project/car-imgs/random/resize/000178.jpg'

    with open("svm_model_HOG_crop.pkl", 'rb') as file:
        clf_grid = pickle.load(file)

    # single_img_check(img_path, img_r_path, eng, clf_grid)

    # train_img_path = '/home/bwijerat/Documents/Thesis/ext-code/Project/car-imgs/train_test2'
    # train_labels = pd.read_csv('/home/bwijerat/Documents/Thesis/ext-code/Project/car-imgs/train_test2.csv')
    #
    # dataset = load_image_desc(train_img_path, train_labels, eng)
    #
    # with open("img_dataset_HOG_crop2.pkl", 'wb') as file:
    #     pickle.dump(dataset, file)

    # with open("img_dataset_HOG_crop.pkl", 'rb') as file:
    #     dataset = pickle.load(file)
    #
    # x_train, x_test, y_train, y_test = train_test_split(
    #         dataset.data, dataset.target, test_size=0.3, random_state=109)
    #
    # # Grid Search: Parameter Grid
    # param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10], 'kernel': ['rbf']}
    #
    # # Make grid search classifier
    # clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
    #
    # # Train the classifier
    # clf_grid.fit(x_train, y_train)
    #
    # y_pred = clf_grid.predict(x_test)
    #
    # # print("Best Parameters:\n", clf_grid.best_params_)
    # # print("Best Estimators:\n", clf_grid.best_estimator_)
    #
    # print("\nClassification report for - \n{}:\n{}\n".format(
    #     clf_grid, metrics.classification_report(y_test, y_pred)))



    input = '/home/bwijerat/Documents/Thesis/ext-code/Project/input'
    output = '/home/bwijerat/Documents/Thesis/ext-code/Project/output'

    imgs_dir = sort_alphanumeric([dI for dI in os.listdir(input) if os.path.isdir(os.path.join(input, dI))])

    dir_num = [dI for dI in os.listdir(output) if os.path.isdir(os.path.join(output, dI))].__len__()
    test_dir = os.path.join(output, 'test_' + str(dir_num))
    os.mkdir(test_dir)

    for dir in imgs_dir:

        output_dir = os.path.join(test_dir, dir)
        os.mkdir(output_dir)

        input_dir = os.path.join(input, dir, 'input2')

        run_object_detection(input_dir, output_dir, eng, clf_grid, IS_OUTPUT=True)


    print(1)