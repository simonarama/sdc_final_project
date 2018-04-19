#!/usr/bin/env python
"""
helper functions used by other files
"""
import rospy
from helpers import TLClassifierTestNode

if __name__ == '__main__':
    try:
        TLClassifierTestNode(
            node_name='tl_classifier_test_real',
            inference_file='../data/models/ssd_mobilenet_v2_tl_real_3_classes.pb',
            label_map_file='../data/labelmaps/annotated_label_map_3.pbtxt',
            num_classes=3,
            input_image_topic='/image_raw',
            output_image_topic='/traffic_light_classification_real'
        )
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
