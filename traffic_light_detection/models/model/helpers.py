"""
helper functions used by other files
"""

import six.moves.urllib as urllib
import tarfile

from os import listdir
from os.path import isfile, join
from PIL import Image

import numpy as np
import tensorflow as tf

def check_tensorflow_version():
    """check the tensorflow version and raise exception when not qualified"""
    if tf.__version__ < '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def load_image(image_path):
    """load the image with the given image path to numpy array"""
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_files_in_folder(folder_path):
    """get all file pathes in the given folder"""
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]


def download_model(download_url, model_file):
    # download the model from the url and save as model_file
    opener = urllib.request.URLopener()
    opener.retrieve(download_url, model_file)
    # unzip the file and get the frozen inference graph
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

def load_tensor_graph_from_checkpoint(checkpoint_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph