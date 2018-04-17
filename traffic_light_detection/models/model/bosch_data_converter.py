import os
import progressbar
import tensorflow as tf
import yaml

from random import shuffle
from object_detection.utils import dataset_util

class DataConverter:
    TRAINGING_DATA_PER = 0.8

    def __init__(self, image_width, image_height, labels):
        self.image_width = image_width
        self.image_height = image_height
        self.labels = labels

    def convert(self, input_yaml_file, training_output_yaml_file, eval_output_yaml_file):
        examples = self.load_input_examples(input_yaml_file)
        shuffle(examples)

        num_traning_examples = int(len(examples) * DataConverter.TRAINGING_DATA_PER)
        training_examples = examples[:num_traning_examples]
        self.save_tensorflow_record(training_examples, training_output_yaml_file)

        eval_examples = examples[num_traning_examples:]
        self.save_tensorflow_record(eval_examples, eval_output_yaml_file)

    def save_tensorflow_record(self, input_examples, output_file):
        progress = progressbar.ProgressBar()
        writer = tf.python_io.TFRecordWriter(output_file)
        for example in progress(input_examples):
            tf_example = self.create_tensorflow_example(example)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print("{} examples are saved to {}".format(len(input_examples), output_file))

    def load_input_examples(self, input_yaml_file):
        """load the examples from the input yaml file"""
        input_examples = yaml.load(open(input_yaml_file, 'rb').read())

        # replace the path to image with absolute path
        dir_path = os.path.dirname(input_yaml_file)
        for input_example in input_examples:
            input_example['path'] = os.path.abspath(os.path.join(dir_path, input_example['path']))

        print("Loaded {} examples".format(len(input_examples)))
        return input_examples

    def create_tensorflow_example(self, input_example):
        xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
        ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
        classes = [] # List of integer class id of bounding box (1 per box)
        classes_text = [] # List of string class name of bounding box (1 per box)
        for box in input_example['boxes']:
            if float(box['x_min']) > float(box['x_max']) or float(box['y_min']) > float(box['y_max']):
                print('Data Error: bounding box wrong value with sample{}'.format(input_example['path']))
                continue

            xmins.append(float(box['x_min']) / float(self.image_width))
            xmaxs.append(float(box['x_max']) / float(self.image_width))
            ymins.append(float(box['y_min']) / float(self.image_height))
            ymaxs.append(float(box['y_max']) / float(self.image_height))

            classes.append(int(self.labels[box['label']]))
            classes_text.append(box['label'].encode())

        image_format = 'png'.encode()
        filename = input_example['path'] # Filename of the image. Empty if image is not from file
        filename = filename.encode()
        with tf.gfile.GFile(input_example['path'], 'rb') as fid:
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


def convert_small_traffic_14():
    image_height = 720
    image_widht = 1280
    labels =  {
        "Green" : 1,
        "Red" : 2,
        "GreenLeft" : 3,
        "GreenRight" : 4,
        "RedLeft" : 5,
        "RedRight" : 6,
        "Yellow" : 7,
        "off" : 8,
        "RedStraight" : 9,
        "GreenStraight" : 10,
        "GreenStraightLeft" : 11,
        "GreenStraightRight" : 12,
        "RedStraightLeft" : 13,
        "RedStraightRight" : 14
    }
    input_yaml = "../data/training/dataset_train_rgb/train.yaml"
    train_output_record = "../data/training/tensorflow_records/rgb_train_14.tfrecord"
    eval_output_record = "../data/training/tensorflow_records/rgb_eval_14.tfrecord"

    data_converter = DataConverter(image_widht, image_height, labels)
    data_converter.convert(input_yaml, train_output_record, eval_output_record)


def convert_small_traffic_4():
    image_height = 720
    image_width = 1280
    labels =  {
        "Green" : 1,
        "Red" : 2,
        "Yellow" : 3,
        "off" : 4
    }
    input_yaml = "../../data/datasets/dataset_train_rgb/train_class_4.yaml"
    train_output_record = "../../data/tensorflow_records/rgb_train_4.tfrecord"
    eval_output_record = "../../data/tensorflow_records/rgb_eval_4.tfrecord"

    data_converter = DataConverter(image_width, image_height, labels)
    data_converter.convert(input_yaml, train_output_record, eval_output_record)


if __name__ == '__main__':
    #convert_small_traffic_14()
    convert_small_traffic_4()