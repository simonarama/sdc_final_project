"""
helper functions used in tests
it has dependency to tensorflow object detections api
"""
import rospy
from cv_bridge import CvBridge

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tl_classifier import TLClassifier
from sensor_msgs.msg import Image as ImageMessage

def load_category_index(label_map_path, num_classes):
    """
    load the category index from the lablemap with the given path
    for example, a cateory index is like the following
    CATEGORORY_INDEX = {
        1 : {'id':1, 'name':'Green'},
        2 : {'id':2, 'name':'Red'},
        3 : {'id':3, 'name':'Yellow'}
    }
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def draw_detection_on_image(image, output_dict, category_index):
    """
    draw the detection results on the image with the given output_dic and category index
    """
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image


class TLClassifierTestNode(object):
    """
    ros node for traffic light classifier testing, it listens to the camera images and publishes the detection result image
    """
    def __init__(self, node_name, inference_file, label_map_file, num_classes, input_image_topic, output_image_topic):
        # create the ros node
        rospy.init_node(node_name)

        self.bridge = CvBridge()
        self.category_index = load_category_index(label_map_file, num_classes)

        # create the traffic light classifier
        self.classifier = TLClassifier(inference_file)

        # create the publisher to publish result out
        self.classification_publisher = rospy.Publisher(output_image_topic, ImageMessage, queue_size=1)

        # listen to color image
        rospy.Subscriber(input_image_topic, ImageMessage, self.image_color_cb, queue_size=1)

        # keep the loop running
        rospy.spin()

    def image_color_cb(self, image_message):
        """callback function when color image arrives"""
        self.process_message(image_message)

    def process_message(self, message):
        """ process the image message, and publish out the detection result as image """
        cv_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
        detection = self.classifier.infer_image(cv_image)
        draw_detection_on_image(cv_image, detection, self.category_index)

        result_message = self.bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
        self.classification_publisher.publish(result_message)
