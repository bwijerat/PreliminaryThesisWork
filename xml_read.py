
######################################################
# Evaluation code
# By: Brian Wijeratne
# Course: EECS6323
#
# Used to create binary maps from xml data outputted by YOLOv3


import os
import re
import cv2
import numpy as np
from optparse import OptionParser
import xml.etree.ElementTree as ET

def read_xml_create_heatmaps(path_file, model):

    path_o, tail = os.path.split(path_file)
    f_name, f_type = tail.split('.')

    if not os.path.exists(os.path.join(path_o, f_name)):
        os.mkdir(os.path.join(path_o, f_name))
        print("Directory ", f_name, " Created ")
    else:
        print("Directory ", f_name, " already exists")

    # test
    if not os.path.exists(os.path.join(path_o, f_name, 'overlay')):
        os.mkdir(os.path.join(path_o, f_name, 'overlay'))
        print("Directory ", 'overlay', " Created ")
    else:
        print("Directory ", 'overlay', " already exists")

    path_out = os.path.join(path_o,f_name)

    tree = ET.parse(path_file)
    root = tree.getroot()

    width = int(root[2].attrib['width'])
    height = int(root[2].attrib['height'])

    for img_data in root[2:]:
        img = np.zeros((height, width), dtype=np.uint8)
        name = img_data.attrib['name']
        path_output = os.path.join(path_out, name)

        for annotation_data in img_data:
            xtl = int(float(annotation_data.attrib['xtl']))
            ytl = int(float(annotation_data.attrib['ytl']))
            xbr = int(float(annotation_data.attrib['xbr']))
            ybr = int(float(annotation_data.attrib['ybr']))

            xdiff = round((xbr - xtl) / 2)
            ydiff = round((ybr - ytl) / 2)
            xcen = round(xtl + xdiff)
            ycen = round(ytl + ydiff)

            cv2.ellipse(img, (xcen, ycen), (xdiff, ydiff), 0, 0, 360, 255, cv2.FILLED)
            #cv2.rectangle(img, (xtl, ytl), (xbr, ybr), 255, cv2.FILLED)

        print(path_output)
        cv2.imwrite(path_output, img)

        # test
        img_test = cv2.imread(os.path.join('/home/bwijerat/Documents/EECS6323/SMILER-v_1.1/Exp_Archive', model, 'input2', name), 0)
        img_result = cv2.addWeighted(img_test, 1, img, 0.4, 0)
        path_output2 = os.path.join(path_out,'overlay',name)
        cv2.imwrite(path_output2, img_result)

# Code taken from stack overflow:
# https://stackoverflow.com/questions/60416494/i-want-to-plot-each-legend-next-to-my-each-plot-in-the-loop-in-python
def sort_alphanumeric(l):
	"""
	Sorts the given iterable in the way that is expected.
    Required arguments:
    l -- The iterable to be sorted.
    """
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanum_key)

def main():
    parser = OptionParser()

    parser.add_option("-d", "--directory",
                      dest="directory",
                      help="Directory with model outputs",
                      type="string",
                      action="store"
                      )
    (options, args) = parser.parse_args()

    path_o = options.directory
    input = sort_alphanumeric([dI for dI in os.listdir(path_o) if os.path.isdir(os.path.join(path_o, dI))])

    for model in input:

        path_in = os.path.join(path_o, model)
        file_in = sort_alphanumeric([f for f in os.listdir(path_in) if os.path.isfile(os.path.join(path_in, f))])

        for file in file_in:

            # XML
            path_file = os.path.join(path_in, file)
            read_xml_create_heatmaps(path_file, model)

if __name__== "__main__":
	main()




