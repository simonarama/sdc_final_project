"""
object detector
"""

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import helpers

class ObjectDetector(object):
    """ObjectDetector class for object detection"""
    def __init__(self, checkpoint_path, labels_path, num_classes):
        """
        checkpoint_path: Path to frozen detection graph. This is the actual model that is used for the object detection.
        labels_path: List of the strings that is used to add correct label for each box.
        """
        self.detection_graph = helpers.load_tensor_graph_from_checkpoint(checkpoint_path)
        self.label_map = label_map_util.load_labelmap(labels_path)

        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.image_tensor, self.output_dict_tensor = ObjectDetector.load_input_output_tensors(self.detection_graph)
        self.sess = tf.Session(graph=self.detection_graph)

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

    def detect_images(self, image_folder_path, num_images):
        """detect randomely selected num_images images from the given image folder"""
        image_paths = helpers.get_files_in_folder(image_folder_path)

        # randomly selecte num_images from the folder
        selected_image_paths = random.sample(image_paths, num_images)

        # load selected images to the numpy array
        images = []
        for image_path in selected_image_paths:
            image = helpers.load_image(image_path)
            images.append(image)

        self.run_inference_for_images(images)

    def detect(self, image_path):
        """detect obeject in the image with the given image_path"""
        image = helpers.load_image(image_path)
        self.run_inference_for_images([image])

    def run_inference_for_images(self, images):
        """infer the images"""
        for image in images:
            # Run inference
            print("Infer on image with size: {} x {}".format(image.shape[0], image.shape[1]))
            output_dict = self.sess.run(self.output_dict_tensor, feed_dict={self.image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(image, output_dict['detection_boxes'],
                output_dict['detection_classes'], output_dict['detection_scores'], self.category_index,
                use_normalized_coordinates=True, line_thickness=8)

            # display the image
            plt.figure()
            plt.imshow(image)
            plt.show()


def detect_general_classes():
    """detect general classes using existing models directly"""
    #checkpoint_path = '../../data/model_zoo/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
    #checkpoint_path = '../../data/model_zoo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    #checkpoint_path = '../../data/model_zoo/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
    checkpoint_path = '../../data/model_zoo/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    labels_path = './labelmaps/mscoco_label_map.pbtxt'
    num_classes = 90

    detector = ObjectDetector(checkpoint_path, labels_path, num_classes)
    detector.detect_images('test_images/', 10)


def detect_traffic_lights():
    """detect the traffic lights usinge tranfer learned model"""
    checkpoint_path = './inference_graph/tf_1.3.0/ssd_mobilenet_v2_tl_real_3_classes.pb'
    labels_path = './labelmaps/annotated_label_map_3.pbtxt'
    num_classes = 3

    detector = ObjectDetector(checkpoint_path, labels_path, num_classes)
    detector.detect_images('test_images/', 10)
    #detector.detect_images('../../data/datasets/dataset_test_rgb/rgb/test/', 10)


if __name__ == '__main__':
    #detect_general_classes()
    detect_traffic_lights()
