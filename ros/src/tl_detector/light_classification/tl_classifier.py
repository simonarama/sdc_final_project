"""
TLClassifier, the class responsible for detecting traffic light with colors
using pretrained convolutional nueral network.
"""
from os import listdir
from os.path import isfile, join

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from styx_msgs.msg import TrafficLight
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TLClassifier(object):
    """
    TLClassifier, the class responsible for detecting traffic light with colors
    using pretrained convolutional nueral network.
    """
    DETECTION_SCORE_THRESHOLD = 0.5

    def __init__(self, checkpoint_path, labels_path, num_classes):
        self.detection_graph = TLClassifier.load_graph_from_checkpoint(checkpoint_path)
        self.label_map = label_map_util.load_labelmap(labels_path)

        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.image_tensor, self.output_dict_tensor = self.load_input_output_tensors(self.detection_graph)
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image
        Param image (cv::Mat): image containing the traffic light
        Return int, ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        output_dict = self.infer_image(image)

        # check the number of red, green and yellow light detections
        num_red_detections = 0
        num_yellow_detections = 0
        num_green_detections = 0
        for detected_class, score in zip(output_dict['detection_classes'], output_dict['detection_scores']):
            if score < TLClassifier.DETECTION_SCORE_THRESHOLD:
                continue
            if detected_class == 1:
                num_green_detections = num_green_detections + 1
            elif detected_class == 2:
                num_red_detections = num_red_detections + 1
            elif detected_class == 3:
                num_yellow_detections = num_yellow_detections + 1

        # determin the traffic light state based on detections
        light_state = TrafficLight.UNKNOWN
        if num_red_detections == 0 and num_green_detections == 0 and num_yellow_detections == 0:
            light_state = TrafficLight.UNKNOWN
        elif num_red_detections >= num_green_detections and num_red_detections >= num_yellow_detections:
            light_state = TrafficLight.RED
        elif num_yellow_detections >= num_red_detections and num_yellow_detections >= num_green_detections:
            light_state = TrafficLight.YELLOW
        elif num_green_detections >= num_red_detections and num_green_detections >= num_yellow_detections:
            light_state = TrafficLight.GREEN

        # print out the detection result
        print "# red lights: {}, green lights: {}, yellow lights: {}".format(num_red_detections, num_green_detections, num_yellow_detections)
        TLClassifier.print_light_state(light_state)

        return light_state

    def infer_image(self, image):
        """
        detect traffic light on the given image and return the detection result
        """
        output_dict = self.sess.run(self.output_dict_tensor, feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict

    def draw_detection_on_image(self, image, output_dict):
        """
        visualization of the results of a detection.
        """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image

    def infer_images(self, images):
        """
        detect traffic light on the given list of images
        """
        for image in images:
            detection = self.infer_image(image)
            result = self.draw_detection_on_image(image, detection)
            plt.figure()
            plt.imshow(result)
            plt.show()

    def test_detection_on_images(self, image_folder_path, num_images):
        """
        test the detection on images in the given image folder path, it will randomly select
        num_images, run the detection and show the result images.
        """
        image_paths = TLClassifier.get_files_in_folder(image_folder_path)
        selected_image_paths = random.sample(image_paths, num_images)

        for image_path in selected_image_paths:
            self.test_detection_on_image(image_path)

    def test_detection_on_image(self, image_path):
        """
        test the detection on image with the given image_path, it will run the detection and show the result.
        """
        image = TLClassifier.load_image(image_path)
        result = self.infer_image(image)
        TLClassifier.display_image(result)

    @staticmethod
    def load_input_output_tensors(graph):
        """ get handles to input and output tensors """
        # get handle to input image tensor
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        # get handle to output tensor dict tensor
        output_dict_tensor = {}
        ops = graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                output_dict_tensor[key] = graph.get_tensor_by_name(tensor_name)

        return image_tensor, output_dict_tensor

    @staticmethod
    def load_graph_from_checkpoint(checkpoint_path):
        """ load tensor flow graph from checkpoint file """
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
        return graph

    @staticmethod
    def load_image(image_path):
        """ load the image with the given image path to numpy array """
        image = Image.open(image_path)
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    @staticmethod
    def get_files_in_folder(folder_path):
        """ get all file pathes in the given folder """
        return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    @staticmethod
    def display_image(image):
        """ use matplotlib to display the given image """
        plt.figure()
        plt.imshow(image)
        plt.show()

    @staticmethod
    def print_light_state(light_state):
        """ print out the light state for debuging purpose """
        if light_state == TrafficLight.RED:
            print "light state: red"
        elif light_state == TrafficLight.YELLOW:
            print "light state: yellow"
        elif light_state == TrafficLight.GREEN:
            print "lgiht state: green"
        else:
            print "light state: unknown"
