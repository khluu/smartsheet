import sys
import os

# Regex
import re

# PArse xml
import xml.etree.ElementTree as ET

import numpy as np

# Load / dump data
import pickle


from PIL import Image
from PIL import ImageDraw
from statistics import median 
import ctypes
import itertools
import numpy as np
import matplotlib.pyplot as plt

import numpy as np


import h5py
import os
import h5py

#Adapted from here: https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py/121308
def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    #print(len(dic[0]))
    #print(dic[0][5])
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)
        h5file.close();


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    if isinstance(dic, list):
        dic = {str(i):v for i,v in enumerate(dic)}
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, list):
            item_as_dict = {str(i):v for i,v in enumerate(item)}
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item_as_dict)
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


################ DANNY's C Code ########
def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

try:
    bez_fit_lib = ctypes.CDLL('./bezier_fit.so')
except OSError:

    print(  "-----ERROR Missing C module -----\n" +
            "Looks like you're missing bezier_fit.so. "+
            "You can compile it for your system by running 'make bezier_fit.so' "+
            "in AL_HTML/react_interface/src/tutors/Stylus/c_modules/src/ "+
            "then copy it into this directory.")
    sys.exit()

BUFFER_SIZE = 2048

point_ptr = (ctypes.c_double * BUFFER_SIZE)()
bezier_ptr = (ctypes.c_double * BUFFER_SIZE)()
mlFeature_ptr = (ctypes.c_double * (BUFFER_SIZE * 5) )()
c_fitCurve = wrap_function(bez_fit_lib,"c_FitCurve",ctypes.c_int,[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]);
c_mlEncode = wrap_function(bez_fit_lib,"c_ML_EncodeCurves",None,[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double)]);

def fitCurve(coords, error=6.0):
    flat_coords = np.array(coords, dtype=np.float64)[:, :2].reshape(-1)
    #print('Done plotting')
    point_ptr[:len(coords) * 2] = flat_coords  # np.array(list(itertools.chain(*coords)),dtype=np.float64)
    n_beziers = c_fitCurve(point_ptr, len(coords), error * error, bezier_ptr)
    c_mlEncode(bezier_ptr, n_beziers, mlFeature_ptr)
    #print(np.array(mlFeature_ptr[:9*n_beziers]).reshape((-1,9)))
    return np.array(mlFeature_ptr[:9*n_beziers]).reshape((-1,9))

########################################

class Extractor(object):

    """Extracts patterns from inkml files."""

    crohme_package = os.path.join('data', 'CROHME_full_v2')
    output_dir = 'outputs'

    versions_available = ['2011', '2012', '2013', '2020', '2021']

    minx = miny = 123123123
    maxx = maxy = -123123123
    status = 1
    # Loads all categories that are available
    def load_categories(self):

        with open('categories.txt', 'r') as desc:

            lines = desc.readlines()
            #print("hi")

            # Removing any whitespace characters appearing in the lines
            #print(lines)
            categories = [{ "name": line.split(":")[0],
                            "classes": line.split(":")[1].strip().split(" ")}
                            for line in lines]

            return categories

    def __init__(self, versions="2013", categories="all"):

        # try:
        #     self.box_size = int(box_size)
        # except ValueError:
        #     print("\n! Box size must be a number!\n")
        #     exit()

        # Load list of possibble categories
        self.categories_available = self.load_categories()

        # Split by '+' delimeters
        versions = versions.split('+')
        categories = categories.split('+')
        print(categories)
        for version in versions:

            if version not in self.versions_available:

                print("\n! This dataset version does not exist!\n")
                exit()

        self.versions = versions
        print(self.versions)

        # Get names of available categories
        category_names = [category["name"] for category in self.categories_available]
        #print(category_names)
        classes = []
        for category in categories:

            if category in category_names:

                category_idx = category_names.index(category)
                # Get classes of corresponding category
                classes += self.categories_available[category_idx]["classes"]

            else:

                print("\n! This category does not exist!\n")
                print("# Possible categories:\n")
                # [print(" ", category["name"]) for category in self.categories_available]
                exit()

        self.categories = categories
        self.classes = classes

        self.train_data = []
        self.test_data = []
        self.validation_data = []

    def bezier(self, status = 1):
        self.status = status
        # Load inkml files
        for version in self.versions:

            #start test
            if version == "2020":
                #print(fitCurve(b))
                data_dir = os.path.join(self.crohme_package, "test")
                train_dir = os.path.join(data_dir, "trainData")
                #test_dir = os.path.join(data_dir, "testDataGT")
                #validation_dir = os.path.join(data_dir, "testData")

                if self.status == 1:
                    self.train_data += self.parse_inkmls(train_dir)
                else:
                    self.parse_inkmls(train_dir)
                #self.test_data += self.parse_inkmls(test_dir)
                #self.validation_data += self.parse_inkmls(validation_dir)

            if version == "2021":
                data_dir = os.path.join(self.crohme_package, "CROHMETEST2_data")
                train_dir = os.path.join(data_dir, "trainData")
                test_dir = os.path.join(data_dir, "testDataGT")
                validation_dir = os.path.join(data_dir, "testData")

                if self.status == 1:
                    self.train_data += self.parse_inkmls(train_dir)
                else:
                    self.parse_inkmls(train_dir)
                #self.test_data += self.parse_inkmls(test_dir)
                #self.validation_data += self.parse_inkmls(validation_dir)

            #end test
            if version == "2011":
                data_dir = os.path.join(self.crohme_package, "CROHME2011_data")
                train_dir = os.path.join(data_dir, "CROHME_training/CROHME_training")
                test_dir = os.path.join(data_dir, "CROHME_testGT")
                validation_dir = os.path.join(data_dir, "CROHME_testGT/CROHME_testGT")

                if self.status == 1:
                    self.train_data += self.parse_inkmls(train_dir)
                    #self.test_data += self.parse_inkmls(test_dir)
                    self.validation_data += self.parse_inkmls(validation_dir)
                else:
                    self.parse_inkmls(train_dir)
                    #self.parse_inkmls(test_dir)
                    self.parse_inkmls(validation_dir)
            if version == "2012":
                data_dir = os.path.join(self.crohme_package, "CROHME2012_data")
                train_dir = os.path.join(data_dir, "trainData")
                test_dir = os.path.join(data_dir, "testData")
                validation_dir = os.path.join(data_dir, "testDataGT")

                if self.status == 1:
                    self.train_data += self.parse_inkmls(train_dir)
                    #self.test_data += self.parse_inkmls(test_dir)
                    self.validation_data += self.parse_inkmls(validation_dir)
                else:
                    self.parse_inkmls(train_dir)
                    # self.parse_inkmls(test_dir)
                    self.parse_inkmls(validation_dir)

            if version == "2013":
                data_dir = os.path.join(self.crohme_package, "CROHME2013_data")
                train_root_dir = os.path.join(data_dir, "TrainINKML")
                train_dir_1 = os.path.join(train_root_dir, "expressmatch")
                train_dir_2 = os.path.join(train_root_dir, "extension")
                train_dir_3 = os.path.join(train_root_dir, "HAMEX")
                train_dir_4 = os.path.join(train_root_dir, "KAIST")
                train_dir_5 = os.path.join(train_root_dir, "MathBrush")
                train_dir_6 = os.path.join(train_root_dir, "MfrDB")

                test_dir = os.path.join(data_dir, "TestINKML")
                validation_dir = os.path.join(data_dir, "TestINKMLGT")

                if self.status == 1:
                    self.train_data += self.parse_inkmls(train_dir_1)
                    self.train_data += self.parse_inkmls(train_dir_2)
                    self.train_data += self.parse_inkmls(train_dir_3)
                    self.train_data += self.parse_inkmls(train_dir_4)
                    self.train_data += self.parse_inkmls(train_dir_5)
                    self.train_data += self.parse_inkmls(train_dir_6)
                    #self.test_data += self.parse_inkmls(test_dir)
                    self.validation_data += self.parse_inkmls(validation_dir)
                else:
                    self.parse_inkmls(train_dir_1)
                    self.parse_inkmls(train_dir_2)
                    self.parse_inkmls(train_dir_3)
                    self.parse_inkmls(train_dir_4)
                    self.parse_inkmls(train_dir_5)
                    self.parse_inkmls(train_dir_6)
                    #self.parse_inkmls(test_dir)
                    self.parse_inkmls(validation_dir)

        return self.train_data, self.test_data, self.validation_data

    def parse_inkmls(self, data_dir_abs_path):
        print(data_dir_abs_path)
        'Accumulates traces_data of all the inkml files\
        located in the specified directory'
        encoded_samples = []
        # classes_rejected = []

        'Check object is a directory'
        if os.path.isdir(data_dir_abs_path):

            for inkml_file in os.listdir(data_dir_abs_path):

                if inkml_file.endswith('.inkml'):
                    inkml_file_abs_path = os.path.join(data_dir_abs_path, inkml_file)

                    #print('Parsing:', inkml_file_abs_path, '...')

                    ' **** Each entry in traces_data represent SEPARATE pattern\
                        which might(NOT) have its label encoded along with traces that it\'s made up of **** '
                    traces_data_curr_inkml = self.get_traces_data(inkml_file_abs_path)

                    # print("CURR")
                    # print(traces_data_curr_inkml)
                    # 'Each entry in patterns_enc is a dictionary consisting of \
                    # pattern_drawn matrix and its label'
                    encoded_samples.append(self.convert_to_bezier(traces_data_curr_inkml))
                    # patterns_enc += ptrns_enc_inkml_curr
                    # classes_rejected += classes_rej_inkml_curr

        return encoded_samples
    
    def convert_to_bezier(self, traces_data):

        # patterns_enc = []

        symbol_encs = []
        # classes_rejected = []

        # symbol_maxDims = []
        #print(self.minx, self.miny)
        xx = []
        yy = []
        minx = 2400000
        miny = 2400000
        maxx = -2400000
        maxy = -2400000
        if self.status == 1:

            for symbol in traces_data['symbols']:
                minx = 2400000
                miny = 2400000
                maxx = -2400000
                maxy = -2400000
                for trace_id in symbol['trace_group']:
                    trace = traces_data['traces'][trace_id]
                    for coord in trace['coords']:
                        #print(coord)
                        minx = min(minx, coord[0])
                        miny = min(miny, coord[1])

                #print(minx,miny,maxx,maxy)
                for trace_id in symbol['trace_group']:
                    trace = traces_data['traces'][trace_id]
                    for coord in trace['coords']:
                        #print(coord)
                        coord[0] -= minx
                        coord[1] -= miny
                for trace_id in symbol['trace_group']:
                    trace = traces_data['traces'][trace_id]
                    for coord in trace['coords']:
                        #print(coord)
                        maxx = max(maxx,coord[0])
                        maxy = max(maxy, coord[1])
                #print(maxx, maxy)
                #print(symbol)
                for trace_id in symbol['trace_group']:
                    trace = traces_data['traces'][trace_id]
                    for coord in trace['coords']:

                        coord[0] /= max(1/100, max(maxx, maxy) / 100)
                        coord[1] /= max(1/100, max(maxx, maxy) / 100)
    
        for symbol in traces_data['symbols']:
            print(symbol)
            stroke_coords = []
            stroke_beziers = []
            stroke_coords_rev = []
            stroke_beziers_rev = []
            try:
                for trace_id in symbol['trace_group']:
                    trace = traces_data['traces'][trace_id]
                    for coord in trace['coords']:
                        xx.append(coord[0])
                        yy.append(coord[1] * -1)
                    stroke_coords.append(np.array(trace['coords'],dtype=np.float32)[:,:2])
                    stroke_beziers.append(fitCurve(trace['coords']))
                    trace_rev = trace['coords'][::-1]
                    stroke_coords_rev.append(np.array(trace_rev, dtype=np.float32)[:,:2])
                    stroke_beziers_rev.append(fitCurve(trace_rev))
            except Exception:
                print("corrupted data3... skipping")
                continue
            
            symbol_enc = {"label": symbol['label'],"coords": stroke_coords, "feat_bez_curves" : stroke_beziers}
            symbol_enc_rev = {"label": symbol['label'], "coords": stroke_coords_rev, "feat_bez_curves": stroke_beziers_rev}
            symbol_encs.append(symbol_enc)
            #symbol_encs.append(symbol_enc_rev)

        print(len(symbol_encs))
        plt.xlim(-10, 100)
        plt.ylim(-100,10)
        #print(stroke_beziers)
        #plt.plot(xx, yy, 'ro')
        #plt.show()
        return symbol_encs
        # medianMaxDim = median(symbol_maxDims)
        # ScaleRatio = box_size/medianMaxDim

        # print("MEDIAN", medianMaxDim)
        
        # for pattern in traces_data:

        #     trace_group = pattern['trace_group']

        #     'mid coords needed to shift the pattern'
        #     min_x, min_y, max_x, max_y = self.get_min_coords(trace_group)

        #     'traceGroup dimensions'
        #     trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x

        #     'shift pattern to its relative position'
        #     shifted_trace_grp = self.shift_trace_grp(trace_group, min_x=min_x, min_y=min_y)

        #     'Interpolates a pattern so that it fits into a box with specified size'
        #     'method: LINEAR INTERPOLATION'
        #     try:
        #         interpolated_trace_grp = self.interpolate(shifted_trace_grp, \
        #                                              trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width, box_size=self.box_size - 1)
        #     except Exception as e:
        #         print(e)
        #         print('This data is corrupted - skipping.')
        #         classes_rejected.append(pattern.get('label'))

        #         continue

        #     'Get min, max coords once again in order to center scaled patter inside the box'
        #     min_x, min_y, max_x, max_y = self.get_min_coords(interpolated_trace_grp)

        #     centered_trace_grp = self.center_pattern(interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=self.box_size)

        #     print(centered_trace_grp)
        #     # 'Center scaled pattern so it fits a box with specified size'
        #     # pattern_drawn = self.draw_pattern(centered_trace_grp, box_size=self.box_size)
        #     # plt.imshow(pattern_drawn, cmap='gray')
        #     # plt.show()

        #     pattern_enc = dict({'features': pattern_drawn, 'label': pattern.get('label')})

        #     # Filter classes that belong to categories selected by the user
        #     if pattern_enc.get('label') in self.classes:

        #         patterns_enc.append(pattern_enc)

        # return patterns_enc, classes_rejected

    # Extracting / parsing tools below
    def get_traces_data(self, inkml_file_abs_path):

        traces_data = []

        tree = ET.parse(inkml_file_abs_path)
        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"
        print(inkml_file_abs_path)
        'Stores traces_all with their corresponding id'
        traces_all = [{'id': trace_tag.get('id'),
                       'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(
                           float(axis_coord) * 10000) \
                                   for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                      else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(
                           float(axis_coord) * 10000) \
                                            for axis_coord in coord.split(' ')] \
                                  for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                      for trace_tag in root.findall(doc_namespace + 'trace')]

        'Sort traces_all list by id to make searching for references faster'
        traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

        'Always 1st traceGroup is a redundant wrapper'
        traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

        if traceGroupWrapper is not None:
            for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

                label = traceGroup.find(doc_namespace + 'annotation').text

                'traces of the current traceGroup'
                traces_curr = []
                for traceView in traceGroup.findall(doc_namespace + 'traceView'):

                    'Id reference to specific trace tag corresponding to currently considered label'
                    traceDataRef = int(traceView.get('traceDataRef'))

                    'Each trace is represented by a list of coordinates to connect'
                    # single_trace = traces_all[traceDataRef]['coords']
                    traces_curr.append(traceDataRef)


                traces_data.append({'label': label, 'trace_group': traces_curr})

        else:
            'Consider Validation data that has no labels'
            [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

        return {"traces" : traces_all, "symbols" : traces_data}

    def get_min_coords(self, trace_group):

        min_x_coords = []
        min_y_coords = []
        max_x_coords = []
        max_y_coords = []

        for trace in trace_group:

            x_coords = [coord[0] for coord in trace]
            y_coords = [coord[1] for coord in trace]

            min_x_coords.append(min(x_coords))
            min_y_coords.append(min(y_coords))
            max_x_coords.append(max(x_coords))
            max_y_coords.append(max(y_coords))

        return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)

    def reset_min(self):
        self.minx = self.miny = 123123123
        self.maxx = self.maxy = -123123123
    def get_min_set(self):
        return self.minx, self.miny, self.maxx, self.maxy

    'shift pattern to its relative position'
    def shift_trace_grp(self, trace_group, min_x, min_y):

        shifted_trace_grp = []

        for trace in trace_group:
            shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in trace]

            shifted_trace_grp.append(shifted_trace)

        return shifted_trace_grp

    'Interpolates a pattern so that it fits into a box with specified size'
    def interpolate(self, trace_group, trace_grp_height, trace_grp_width, box_size):

        interpolated_trace_grp = []

        if trace_grp_height == 0:
            trace_grp_height += 1
        if trace_grp_width == 0:
            trace_grp_width += 1

        '' 'KEEP original size ratio' ''
        trace_grp_ratio = (trace_grp_width) / (trace_grp_height)

        scale_factor = 1.0
        '' 'Set \"rescale coefficient\" magnitude' ''
        if trace_grp_ratio < 1.0:

            scale_factor = (box_size / trace_grp_height)
        else:

            scale_factor = (box_size / trace_grp_width)

        for trace in trace_group:
            'coordintes convertion to int type necessary'
            interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]

            interpolated_trace_grp.append(interpolated_trace)

        return interpolated_trace_grp

    def center_pattern(self, trace_group, max_x, max_y, box_size):

        x_margin = int((box_size - max_x) / 2)
        y_margin = int((box_size - max_y) / 2)

        return self.shift_trace_grp(trace_group, min_x= -x_margin, min_y= -y_margin)

    def draw_pattern(self, trace_group, box_size):

        pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
        for trace in trace_group:

            ' SINGLE POINT TO DRAW '
            if len(trace) == 1:
                x_coord = trace[0][0]
                y_coord = trace[0][1]
                pattern_drawn[y_coord, x_coord] = 0.0

            else:
                ' TRACE HAS MORE THAN 1 POINT '

                'Iterate through list of traces endpoints'
                for pt_idx in range(len(trace) - 1):

                    'Indices of pixels that belong to the line. May be used to directly index into an array'
                    # pattern_drawn[line(r0=trace[pt_idx][1], c0=trace[pt_idx][0],
                    #                  r1=trace[pt_idx + 1][1], c1=trace[pt_idx + 1][0])] = 0.0
                    img = Image.fromarray(pattern_drawn)
                    draw = ImageDraw.Draw(img)
                    draw.line([(trace[pt_idx][0], trace[pt_idx][1]), (trace[pt_idx + 1][0], trace[pt_idx + 1][1])], fill=0, width=3)

                    pattern_drawn = np.array(img)

        return pattern_drawn

# Converts label to one-hot format
def to_one_hot(class_name, classes):

    one_hot = np.zeros(shape=(len(classes)), dtype=np.int8)
    class_index = classes.index(class_name)
    one_hot[class_index] = 1

    return one_hot

def save_data(datas):
    count = 0
    for i, data in enumerate(datas):
        #DO SOMETING
        pass
        # for point in data:
            # if point["label"] == "/":
            #     point["label"] = "forward-slash"
            # if not os.path.exists("extracted_images/" + point["label"]):
            #     print("new label", point["label"])
            #     os.makedirs("extracted_images/" + point["label"])
            # point["features"] = point["features"] * 255
            # point["features"] = point["features"].astype(np.uint8)
            # Image.fromarray(point["features"]).convert("RGB").save("extracted_images/%s/%d_%d.png" % (point["label"], count, i))
            # count += 1

if __name__ == '__main__':

    out_formats = ['pixels', 'hog', 'phog']

    if len(sys.argv) < 1:

        print("\n! Usage:", "python", sys.argv[0], "<dataset_version=2013>", "<category=all>\n")
        exit()

    elif len(sys.argv) >= 1:

        # if sys.argv[1] in out_formats:

        #     out_format = sys.argv[1]
        #     extractor = Extractor(sys.argv[2])
        # else:

        #     print("\n! This output format does not exist!\n")
        #     print("# Possible output formats:\n")
        #     # [print(" ", out_format) for out_format in out_formats]
        #     exit()
        if len(sys.argv) == 2:
            extractor = Extractor(sys.argv[1])
        elif len(sys.argv) == 3:
            extractor = Extractor(sys.argv[1], sys.argv[2])

    # Extract pixel features
    # if out_format == out_formats[0]:
    #extractor.bezier(0)
    #print(extractor.get_min_set())
    train_data, test_data, validation_data = extractor.bezier(1)
    save_dict_to_hdf5(train_data, 'train.hdf5')
    #save_dict_to_hdf5(test_data, 'test.hdf5')
    #save_dict_to_hdf5(validation_data, 'validation.hdf5')
    # save_dict_to_hdf5({str(i):v for i,v in enumerate(train_data)}, 'train.hdf5')
    # save_dict_to_hdf5({str(i):v for i,v in enumerate(test_data)}, 'test.hdf5')
    # save_dict_to_hdf5({str(i):v for i,v in enumerate(validation_data)}, 'validation.hdf5')
    # data_to_save = [train_data.copy(), test_data.copy(), validation_data.copy()]
    # print(train_data)
    # print(test_data)
    # print(validation_data)
    # Get list of all classes
    # classes = sorted(list(set([data_record['label'] for data_record in train_data+test_data])))
    # print()
    # print('How many classes:', len(classes))

    # raise NotImplemented("I haven't actually written any thing to store or use this dataset"
    #                      "yet. Might want to store as an HDF5 and load it into a notebook.")
    # with open('classes.txt', 'w') as desc:
    #     for r_class in classes:
    #         desc.write(r_class + '\n')

    # ### Save DATA new ###
    #     if not os.path.exists("extracted_images"):
    #         os.makedirs("extracted_images")
    #     save_data([train_data, test_data, validation_data])

    # print("ALL DONE")


        ###
        # 1. Flatten image to single feaute map (vector of pixel intensities)
        # 2. Convert its label to one-hot format
    #     train_data = [{'label': to_one_hot(train_rec['label'], classes), 'features': train_rec['features'].flatten()} for train_rec in train_data]
    #     test_data = [{'label': to_one_hot(test_rec['label'], classes), 'features': test_rec['features'].flatten()} for test_rec in test_data]
    #     validation_data = [{'label': to_one_hot(validation_rec['label'], classes), 'features': validation_rec['features'].flatten()} for validation_rec in validation_data]

    # # Extract HOG features
    # elif out_format == out_formats[1]:
    #     train_data, test_data, validation_data = extractor.hog()

    # # Extract PHOG features
    # elif out_format == out_formats[2]:
    #     train_data, test_data, validation_data = extractor.phog()

    # output_dir = os.path.abspath(extractor.output_dir)
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    # train_out_dir = os.path.join(output_dir, 'train')
    # test_out_dir = os.path.join(output_dir, 'test')
    # validation_out_dir = os.path.join(output_dir, 'validation')

    # # Save data
    # print('\nDumping extracted data ...')
    # # Make directories if needed
    # if not os.path.exists(train_out_dir):
    #     os.mkdir(train_out_dir)
    # if not os.path.exists(test_out_dir):
    #     os.mkdir(test_out_dir)
    # if not os.path.exists(validation_out_dir):
    #     os.mkdir(validation_out_dir)

    # with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as train:
    #     pickle.dump(train_data, train, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('Data has been successfully dumped into', train.name)

    # with open(os.path.join(test_out_dir, 'test.pickle'), 'wb') as test:
    #     pickle.dump(test_data, test, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('Data has been successfully dumped into', test.name)

    # with open(os.path.join(validation_out_dir, 'validation.pickle'), 'wb') as validation:
    #     pickle.dump(validation_data, validation, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('Data has been successfully dumped into', validation.name)
