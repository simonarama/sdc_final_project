#!/usr/bin/env python
"""
helper functions used by other files
"""

import rospy

from sensor_msgs.msg import Image as ImageMessage
from cv_bridge import CvBridge

from light_classification.tl_classifier import TLClassifier

class TLClassifierTestNodeReal(object):
    """
    ros node for traffic light classifier testing, it listens to the camera images and publishes the detection result image
    """
    def __init__(self):
        rospy.init_node('tl_classifier_test')
        self.bridge = CvBridge()
        # create the traffic light classifier
        inference_file = '../../data/models/ssd_mobilenet_v2_traffic_light_inference_graph.pb'
        labels_file ='../../data/small_traffic_label_map_4.pbtxt'
        num_classes = 4
        self.classifier = TLClassifier(inference_file, labels_file, num_classes)
        # create the publisher to publish result out
        self.classification_publisher = rospy.Publisher('/traffic_light_classification', ImageMessage, queue_size=1)
        # listen to raw image published by testing ros bag file from course
        rospy.Subscriber('/image_raw', ImageMessage, self.image_color_cb, queue_size=1)
        # keep the loop running
        rospy.spin()

    def image_color_cb(self, image_message):
        """ callback function when color image arrives """
        self.process_message(image_message)

    def process_message(self, message):
        """ process the image message, and publish out the detection result as image """
        cv_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
        detection = self.classifier.infer_image(cv_image)
        self.classifier.draw_detection_on_image(cv_image, detection)

        result_message = self.bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
        self.classification_publisher.publish(result_message)


if __name__ == '__main__':
    try:
        TLClassifierTestNodeReal()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
