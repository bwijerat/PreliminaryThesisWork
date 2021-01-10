
######################################################
# Evaluation code
# By: Brian Wijeratne
# Course: EECS6323
#
# Used to post process heatmaps from SMILER framework, setup evaluation of dominant salient areas against Grount Truth


import os
import re
import cv2
import numpy as np
from collections import Counter
from optparse import OptionParser
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_uint
import copy


def connected_components(path1, path2, input_names, file_names):
	dirName = 'connect'

	if not os.path.exists(os.path.join(path2, dirName)):
		os.mkdir(os.path.join(path2, dirName))
		print("Directory ", dirName, " Created ")
	else:
		print("Directory ", dirName, " already exists")

	orb = cv2.ORB_create()
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	count = 0
	MAX_AREAS = 10
	DONT_SKIP = False

	# track previous frame info
	img_roi2_list = []
	roi2_label_list = []
	roi2_pixel_lists = []
	roi2_area_list = []

	for f in file_names:

		i = input_names[count]
		count += 1

		f_name, f_type = f.split('.')
		path3 = os.path.join(path2, dirName, f_name + '-connect.jpg')

		if not os.path.isfile(path3):
			img = cv2.imread(os.path.join(path2, f), 0)
			IMG = cv2.imread(os.path.join(path1, i), 0)

			# OPERATION

			# track current frame info
			img_roi1_list = []
			roi1_pixel_lists = []
			roi1_area_list = []
			roi1_xywh_list = []

			binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			# getting contours regions
			contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contours = contours[0]

			# remove any small components
			xywh_temp = []
			ii = -1
			for cnt in contours:
				ii += 1
				x1, y1, w1, h1 = cv2.boundingRect(cnt)
				cnt_len = cv2.arcLength(cnt, True)	# run something before if statement to successfully run if statement
				xywh_temp.append([x1, y1, w1, h1])
			ii = -1
			for xywh in xywh_temp:
				ii += 1
				x1, y1, w1, h1 = xywh
				if (w1 < 32) or (h1 < 32):
					del contours[ii]
					ii -= 1
					binary[y1:y1 + h1, x1:x1 + w1] = 0

			# getting mask with connectComponents
			ret, labels = cv2.connectedComponents(binary)

			# Map component labels to hue val
			label_hue = copy.deepcopy(np.uint8(179 * labels / MAX_AREAS))
			blank_ch = 255 * np.ones_like(label_hue)
			labeled_img = copy.deepcopy(cv2.merge([label_hue, blank_ch, blank_ch]))

			# cvt to BGR for display
			labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

			# set bg label to black
			labeled_img[label_hue == 0] = 0
			cv2.imshow('labeled_curr.png', labeled_img)

			for cnt in contours:

				# Region of Interest mask
				(x1, y1, w1, h1) = cv2.boundingRect(cnt)
				label_ROI = copy.deepcopy(labels[y1:y1 + h1, x1:x1 + w1])

				# find label value of current contour
				hist, bins = np.histogram(label_ROI, ret + 1, [0, ret + 1])
				hist_sort = sorted(range(len(hist)), key=lambda k: hist[k], reverse=True)
				if (hist_sort[0] != 0):
					label_value = hist_sort[0]
				else:
					label_value = hist_sort[1]

				label_ROI[label_ROI != label_value] = 0
				label_ROI[label_ROI == label_value] = 1
				label_ROI = np.uint8(label_ROI)

				# Apply mask
				img_roi1 = copy.deepcopy(IMG[y1:y1 + h1, x1:x1 + w1])
				img_roi1 *= label_ROI

				# Global values for non-zero ROI pixels
				indices = np.nonzero(label_ROI)
				indexes = zip(indices[0] + y1, indices[1] + x1)

				img_roi1_list.append(img_roi1)
				roi1_pixel_lists.append(indexes)
				roi1_area_list.append(w1*h1)
				roi1_xywh_list.append([x1, y1, w1, h1])

			if DONT_SKIP:
				i_cnt = -1
				for img_roi1 in img_roi1_list:
					i_cnt += 1
					j_cnt = -1

					match_pixels = []

					roi1_pixel_list = roi1_pixel_lists[i_cnt]

					for img_roi2 in img_roi2_list:
						j_cnt += 1

						roi2_pixel_list = roi2_pixel_lists[j_cnt]

						matched_num = set(roi1_pixel_list).intersection(roi2_pixel_list).__len__()
						match_pixels.append(matched_num)


					match_sorted_idx = sorted(range(len(match_pixels)), key=lambda k: match_pixels[k], reverse=True)

					nonzero_idx = []
					for idx in match_sorted_idx:
						if match_pixels[idx] > 0:
							nonzero_idx.append(idx)

					if nonzero_idx.__len__() > 0:

						max_idx = nonzero_idx[0]

						if nonzero_idx.__len__() > 1:
							for idx in nonzero_idx[1:]:
								area1 = roi2_area_list[max_idx]
								area2 = roi2_area_list[idx]
								if area2 > area1:
									max_idx = idx

						(x1, y1, w1, h1) = roi1_xywh_list[i_cnt]
						label_ROI = labels[y1:y1 + h1, x1:x1 + w1]

						# find label value of current roi1 contour
						hist, bins = np.histogram(label_ROI, ret + 1, [0, ret + 1])
						hist_sort = sorted(range(len(hist)), key=lambda k: hist[k], reverse=True)
						if (hist_sort[0] != 0):
							roi1_label = hist_sort[0]
						else:
							roi1_label = hist_sort[1]

						# roi2 and roi1 labels values
						label_value_roi2 = roi2_label_list[max_idx]
						label_value_roi1 = roi1_label

						print('\nlabel_value_roi2: ', label_value_roi2)
						print('label_value_roi1: ', label_value_roi1)
						cv2.imshow('roi1', img_roi1)
						cv2.imshow('roi2', img_roi2_list[max_idx])

						# remap roi1 to roi2 label value (use 100 as placeholder)
						if label_value_roi1 != label_value_roi2:

							labels[labels == label_value_roi2] = 100
							labels[labels == label_value_roi1] = label_value_roi2
							labels[labels == 100] = label_value_roi1

							# Map component labels to hue val
							label_hue = copy.deepcopy(np.uint8(179 * labels / MAX_AREAS))
							blank_ch = 255 * np.ones_like(label_hue)
							labeled_img = copy.deepcopy(cv2.merge([label_hue, blank_ch, blank_ch]))

							# cvt to BGR for display
							labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

							# set bg label to black
							labeled_img[label_hue == 0] = 0
							cv2.imshow('labeled_change.png', labeled_img)
							pass

			# dont skip comparing with the previous frame
			DONT_SKIP = True

			# track previous frame info
			img_roi2_list = copy.deepcopy(img_roi1_list)
			roi2_pixel_lists = copy.deepcopy(roi1_pixel_lists)
			roi2_area_list = copy.deepcopy(roi1_area_list)

			roi2_label_list = []
			for cnt in contours:

				# Region of Interest mask
				(x1, y1, w1, h1) = cv2.boundingRect(cnt)
				label_ROI = copy.deepcopy(labels[y1:y1 + h1, x1:x1 + w1])

				# find label value of current contour
				hist, bins = np.histogram(label_ROI, ret + 1, [0, ret + 1])
				hist_sort = sorted(range(len(hist)), key=lambda k: hist[k], reverse=True)
				if (hist_sort[0] != 0):
					label_value = hist_sort[0]
				else:
					label_value = hist_sort[1]

				roi2_label_list.append(label_value)


			# Map component labels to hue val
			label_hue = copy.deepcopy(np.uint8(179 * labels / MAX_AREAS))
			blank_ch = 255 * np.ones_like(label_hue)
			labeled_img = copy.deepcopy(cv2.merge([label_hue, blank_ch, blank_ch]))

			# cvt to BGR for display
			labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

			# set bg label to black
			labeled_img[label_hue == 0] = 0

			cv2.imshow('labeled_prev.png', labeled_img)

			print(path3)
			cv2.imwrite(path3, labeled_img)
		else:
			print('(Skipped) ' + path3)

def overlay(path1, path2, input_names, file_names):

	dirName = 'overlay'

	if not os.path.exists(os.path.join(path2, dirName)):
		os.mkdir(os.path.join(path2, dirName))
		print("Directory ", dirName, " Created ")
	else:
		print("Directory ", dirName, " already exists")

	count = 0

	for f in file_names:

		i = input_names[count]
		count += 1

		f_name, f_type = f.split('.')
		path3 = os.path.join(path2, dirName, f_name + '-overlay.jpg')

		if not os.path.isfile(path3):

			img_i = cv2.imread(os.path.join(path1, i), 3)
			img_f = cv2.imread(os.path.join(path2, f), 3)

			# Operation
			result = cv2.addWeighted(img_i, 1, img_f, 0.4, 0)

			print(path3)
			cv2.imwrite(path3, result)
		else:
			print('(Skipped) ' + path3)

def threshold(path2, file_names):

	dirName = 'thresh'

	if not os.path.exists(os.path.join(path2, dirName)):
		os.mkdir(os.path.join(path2, dirName))
		print("Directory ", dirName, " Created ")
	else:
		print("Directory ", dirName, " already exists")

	for f in file_names:

		f_name, f_type = f.split('.')
		path3 = os.path.join(path2, dirName, f_name + '-thresh.jpg')

		if not os.path.isfile(path3):
			img = cv2.imread(os.path.join(path2, f), 0)

			# Operation
			ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			print(path3)
			cv2.imwrite(path3, th3)
		else:
			print('(Skipped) ' + path3)

def cubic(path2, file_names):

	dirName = 'cubic'

	if not os.path.exists(os.path.join(path2, dirName)):
		os.mkdir(os.path.join(path2, dirName))
		print("Directory ", dirName, " Created ")
	else:
		print("Directory ", dirName, " already exists")

	for f in file_names:

		f_name, f_type = f.split('.')
		path3 = os.path.join(path2, dirName, f_name + '-cubic.jpg')

		if not os.path.isfile(path3):
			img = cv2.imread(os.path.join(path2, f), 0)

			# Operation
			img_1 = np.power(img_as_float(img), 3)
			img_2 = np.floor(img_1 * (255.0 / img_1.max()))

			print(path3)
			cv2.imwrite(path3, img_2)
		else:
			print('(Skipped) ' + path3)

def histo(path2, file_names):

	dirName = 'histo/'

	if not os.path.exists(path2 + dirName):
		os.mkdir(path2 + dirName)
		print("Directory " , dirName ,  " Created ")
	else:    
		print("Directory " , dirName ,  " already exists")

	for f in file_names:

		f_name, f_type = f.split('.')
		path3 = path2 + dirName + f_name + '-histo.jpg'

		if not os.path.isfile(path3):
			img = cv2.imread(path2 + f, 0)

			# Operation
			equ = cv2.equalizeHist(img)

			print(path3)
			cv2.imwrite(path3, equ)
		else:
			print('(Skipped) ' + path3)

def sort_alphanumeric(l):
	""" Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanum_key)

def input2_rename(path1, path2, input_names):

	dirName = 'input2'

	if not os.path.exists(os.path.join(path2, dirName)):
		os.mkdir(os.path.join(path2, dirName))
		print("Directory ", dirName, " Created ")
	else:
		print("Directory ", dirName, " already exists")

	count = 0

	for f in input_names:

		f_name, f_type = f.split('.')
		f_name_ = f_name.split('-')

		f_temp = 'frame_%05d' % count
		for ff in f_name_[1:]:
			f_temp += '-' + ff
		f_name = f_temp

		path3 = os.path.join(path2, dirName, f_name + '.jpg')
		count += 1

		if not os.path.isfile(path3):

			print(path3)
			cv2.imwrite(path3, cv2.imread(os.path.join(path1, f), 3))
		else:
			print('(Skipped) ' + path3)


def main():
	
	parser = OptionParser()
	
	parser.add_option("-d", "--directory",
		        dest = "directory",
		        help = "Directory with model outputs",
		        type = "string",
		        action = "store"
		        )
	(options, args) = parser.parse_args()

	# parsing SMILER output directories
	path_o = options.directory
	output = sort_alphanumeric( [dI for dI in os.listdir(path_o) if os.path.isdir(os.path.join(path_o,dI))] )

	# parsing SMILER input directory
	path_i, tail = os.path.split(path_o)
	path_in = os.path.join(path_i,'input')
	included_extensions = ['jpg']
	input_names_sorted = sort_alphanumeric( [fn for fn in os.listdir(path_in)
									  if any(fn.endswith(ext) for ext in included_extensions)] )

	output = ['DGII']
	for model in output:

		#histo(path2, file_names_sorted)

		# CUBIC
		path_out = os.path.join(path_o, model)
		#file_names_sorted = sort_alphanumeric([fn for fn in os.listdir(path_out) if any(fn.endswith(ext) for ext in included_extensions)])
		#cubic(path_out, file_names_sorted)

		# THRESHOLD
		path_out = os.path.join(path_o, model, 'cubic')
		#file_names_sorted = sort_alphanumeric([fn for fn in os.listdir(path_out) if any(fn.endswith(ext) for ext in included_extensions)])
		#threshold(path_out, file_names_sorted)

		# OVERLAY
		#path_out = os.path.join(path_o, model, 'cubic', 'thresh')
		path_out = os.path.join(path_o, model, 'cubic', 'thresh','connect')
		file_names_sorted = sort_alphanumeric([fn for fn in os.listdir(path_out) if any(fn.endswith(ext) for ext in included_extensions)])
		overlay(path_in, path_out, input_names_sorted, file_names_sorted)

		#connected_components(path_in, path_out, input_names_sorted, file_names_sorted)

		#input2_rename(path_in, path_i, input_names_sorted)



  
if __name__== "__main__":
	main()
