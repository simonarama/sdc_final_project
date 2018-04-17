import os
import tensorflow as tf
import progressbar
import yaml

import helpers
import Image
import ImageDraw

from os import listdir
from os.path import isfile, join

from random import shuffle
from object_detection.utils import dataset_util
import matplotlib.pyplot as plt

class AnnotatedDataConverter:
    """ class to convert the annotated data to tf record file

    example of the input annotated data sample is as following
    - annotations:
        - {class: Green, x_width: 52.65248226950354, xmin: 130.4964539007092, y_height: 119.60283687943263, ymin: 289.36170212765956}
        - {class: Green, x_width: 50.156028368794296, xmin: 375.60283687943263, y_height: 121.87234042553195, ymin: 293.90070921985813}
        - {class: Green, x_width: 53.33333333333326, xmin: 623.6595744680851, y_height: 119.82978723404256, ymin: 297.7588652482269}
      class: image
      filename: sim_data_capture/left0003.jpg
    """
    TRAINGING_DATA_PER = 0.85

    def __init__(self, image_width, image_height, labels):
        self.image_width = image_width
        self.image_height = image_height
        self.labels = labels

    def convert(self, input_yaml_file, training_output_yaml_file, eval_output_yaml_file):
        """ convert the given input yaml file to output file for both training and evaluation """
        examples = AnnotatedDataConverter.load_input_examples(input_yaml_file)
        self.convert_examples(examples, training_output_yaml_file, eval_output_yaml_file)

    def convert_with_nolight_images(self, input_yaml_file, nolight_folder, training_output_yaml_file, eval_output_yaml_file):
        """ convert the given input yaml file and a folder of nolight images to output file for both training and evaluation """
        examples = AnnotatedDataConverter.load_input_examples(input_yaml_file)
        no_class_examples = self.load_no_class_examples(nolight_folder)
        all_examples = examples + no_class_examples
        self.convert_examples(all_examples, training_output_yaml_file, eval_output_yaml_file)

    def load_no_class_examples(self, image_folder):
        """ load the examples with no class from the image folder """
        examples = []
        image_paths = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
        for image_path in image_paths:
            example = {}
            example['annotations'] = []
            example['filename'] = os.path.abspath(os.path.join(image_folder, image_path))
            examples.append(example)
        print "loaded {} no light examples".format(len(examples))
        return examples

    def convert_examples(self, examples, training_output_yaml_file, eval_output_yaml_file):
        """convert the given examples to output file for both training and evaluation """
        # shuffle all examples in the dataset
        shuffle(examples)

        #AnnotatedDataConverter.browse_examples(examples)

        #split shuffed samples to training and evaluation dataset
        num_traning_examples = int(len(examples) * AnnotatedDataConverter.TRAINGING_DATA_PER)
        training_examples = examples[:num_traning_examples]
        self.save_tensorflow_record(training_examples, training_output_yaml_file)

        eval_examples = examples[num_traning_examples:]
        self.save_tensorflow_record(eval_examples, eval_output_yaml_file)

    @staticmethod
    def load_input_examples(input_yaml_file):
        """ load the examples from the input yaml file and replace file path with absolute file path """
        input_examples = yaml.load(open(input_yaml_file, 'rb').read())

        # replace the path to image with absolute path
        dir_path = os.path.dirname(input_yaml_file)
        for input_example in input_examples:
            input_example['filename'] = os.path.abspath(os.path.join(dir_path, input_example['filename']))
            #print "sample path: {}".format(input_example['filename'])

        print "Loaded {} examples".format(len(input_examples))
        return input_examples

    @staticmethod
    def browse_examples(input_examples):
        """ browse through the examples, draw boxes on top and check if boxes are right """
        count = 1
        for input_example in input_examples:
            image_file_path = input_example['filename']
            print 'check file {} with path {}'.format(count, image_file_path)
            count += 1
            image = Image.open(image_file_path)
            draw = ImageDraw.Draw(image)
            for box in input_example['annotations']:
                x_min = float(box['xmin'])
                y_min = float(box['ymin'])
                width = float(box['x_width'])
                height = float(box['y_height'])
                color = (255, 255, 255, 255)
                if box['class'] == 'Red':
                    color = (255, 0, 0, 255)
                elif box['class'] == 'Green':
                    color = (0, 255, 0, 255)
                elif box['class'] == 'Yellow':
                    color =(255, 255, 0, 255)
                draw.rectangle([(x_min, y_min), (x_min + width, y_min + height)], outline=color)
            plt.figure(figsize=(16,12))
            plt.imshow(image)
            plt.show()

    def save_tensorflow_record(self, input_examples, output_file):
        """ save the examples to the output_file as tensorflow records """
        progress = progressbar.ProgressBar()
        writer = tf.python_io.TFRecordWriter(output_file)
        for example in progress(input_examples):
            tf_example = self.create_tensorflow_example(example)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print "{} examples are saved to {}".format(len(input_examples), output_file)

    def create_tensorflow_example(self, input_example):
        """ create a tensorflow record from the given example """
        xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
        ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
        classes = [] # List of integer class id of bounding box (1 per box)
        classes_text = [] # List of string class name of bounding box (1 per box)
        for box in input_example['annotations']:
            x_min = float(box['xmin']) / float(self.image_width)
            x_max = (float(box['xmin']) + float(box['x_width'])) / float(self.image_width)
            y_min = float(box['ymin']) / float(self.image_height)
            y_max = (float(box['ymin']) + float(box['y_height'])) / float(self.image_height)

            xmins.append(x_min)
            xmaxs.append(x_max)
            ymins.append(y_min)
            ymaxs.append(y_max)
            classes.append(int(self.labels[box['class']]))
            classes_text.append(box['class'].encode())

        image_format = 'jpg'.encode()
        filename = input_example['filename'] # Filename of the image. Empty if image is not from file
        filename = filename.encode()
        with tf.gfile.GFile(input_example['filename'], 'rb') as fid:
            encoded_image = fid.read()

        # create the tensorflow example based on input example data
        tensorflow_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(self.image_height),
            'image/width': dataset_util.int64_feature(self.image_width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tensorflow_example

def convert_annotated_sim_data():
    image_height = 600
    image_width = 800
    labels = {
        "Green" : 1,
        "Red" : 2,
        "Yellow" : 3
    }
    data_converter = AnnotatedDataConverter(image_width, image_height, labels)

    input_yaml = "../../data/datasets/dataset-sdcnd-capstone/data/sim_training_data/sim_data_annotations.yaml"
    train_output_record = "../../data/tensorflow_records/annotated_sim_train.tfrecord"
    eval_output_record = "../../data/tensorflow_records/annotated_sim_eval.tfrecord"
    data_converter.convert(input_yaml, train_output_record, eval_output_record)


def convert_annotated_real_data():
    image_height = 1096
    image_width = 1368
    labels = {
        "Green" : 1,
        "Red" : 2,
        "Yellow" : 3
    }
    data_converter = AnnotatedDataConverter(image_width, image_height, labels)

    input_yaml = "../../data/datasets/dataset-sdcnd-capstone/data/real_training_data/real_data_annotations.yaml"
    no_light_examples_folder = "../../data/datasets/dataset-sdcnd-capstone/data/real_training_data/nolight"
    train_output_record = "../../data/tensorflow_records/annotated_real_train.tfrecord"
    eval_output_record = "../../data/tensorflow_records/annotated_real_eval.tfrecord"
    data_converter.convert_with_nolight_images(input_yaml, no_light_examples_folder, train_output_record, eval_output_record)


if __name__ == '__main__':
    convert_annotated_real_data()
