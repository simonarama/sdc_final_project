#!/usr/bin/env python
"""
helper functions used by other files
"""

import rospy

from sensor_msgs.msg import Image as ImageMessage
from cv_bridge import CvBridge

from light_classification.tl_classifier import TLClassifier
import settings

class TLClassifierTestNode(object):
    """
    ros node for traffic light classifier testing, it listens to the camera images and publishes the detection result image
    """
    def __init__(self):
        rospy.init_node('tl_classifier_test')
        self.bridge = CvBridge()
        # create the traffic light classifier
        self.classifier = TLClassifier(settings.traffic_light_inference_file, settings.traffic_light_labels_file)
        # create the publisher to publish result out
        self.classification_publisher = rospy.Publisher('/traffic_light_classification', ImageMessage, queue_size=1)
        # listen to color image
        rospy.Subscriber('/image_color', ImageMessage, self.image_color_cb, queue_size=1)
        # keep the loop running
        rospy.spin()

    def image_color_cb(self, image_message):
        """callback function when color image arrives"""
        self.process_message(image_message)

    def process_message(self, message):
        """ process the image message, and publish out the detection result as image """
        cv_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
        result = self.classifier.infer_image(cv_image)

        result_message = self.bridge.cv2_to_imgmsg(result, encoding="rgb8")
        self.classification_publisher.publish(result_message)


if __name__ == '__main__':
    try:
        TLClassifierTestNode()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
