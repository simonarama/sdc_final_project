#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image

from common.waypoints import WayPoints

import tf
import cv2
import yaml

from cv_bridge import CvBridge
import settings
from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        # init member variables
        self.pose = None
        self.waypoints = WayPoints()
        self.camera_image = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.has_image = False
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.light_classifier = TLClassifier(
            settings.traffic_light_inference_file,
            settings.traffic_light_labels_file,
            settings.traffic_light_num_classes)

        self.lights = []
        # init ros node
        rospy.init_node('tl_detector')

        # subscribe to interesting topics
        self.subscribe_to_topics()

        # load traffic light configuration file
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # create red light pubilisher
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # enter the ros loop
        rospy.spin()

    def subscribe_to_topics(self):
        """ subscribe to intersting topics """
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        """
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        """
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

    def pose_cb(self, msg):
        """ callback on topic /current_pose"""
        self.pose = msg

    def waypoints_cb(self, waypoints):
        """ callback on topic /base_waypoints"""
        self.waypoints.set_waypoints(waypoints)

    def traffic_cb(self, msg):
        """ callback on topic /vehicle/traffic_lights"""
        self.lights = msg.lights

    def image_cb(self, msg):
        """
        Identifies red lights in the incoming camera image and publishes the index
        of the waypoint closest to the red light's stop line to /traffic_waypoint
        Param: msg (Image), image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        # used only in simulator, ground truth data is got from simulator
        # light_wp, state = self.process_traffic_lights_sim()

        self.update_state(light_wp, state)

    def process_traffic_lights(self):
        """
        Finds closest visible traffic light, if one exists, and determines its location and color
        Return:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        line_wp_idx = self.find_closest_traffic_light()
        if line_wp_idx is None:
            return -1, TrafficLight.UNKNOWN

        light_state = self.get_light_state()
        return line_wp_idx, light_state

    def find_closest_traffic_light(self):
        """
        find the stop line waypoint index of the closest traffic light to current pose
        """
        line_wp_idx = None
        if self.pose:
            car_wp_idx = self.waypoints.get_closest_waypoint_idx(self.pose.pose.position.x, self.pose.pose.position.y)
            # find the closest visible traffic light (if one exists)
            min_distance = self.waypoints.get_num_waypoints()
            stop_line_positions = self.config['stop_line_positions']
            for line in stop_line_positions:
                # get stop line waypoint index
                temp_wp_idx = self.waypoints.get_closest_waypoint_idx(line[0], line[1])
                # find closest stop line waypoint index
                distance = temp_wp_idx - car_wp_idx
                if distance >= 0 and distance < min_distance:
                    min_distance = distance
                    line_wp_idx = temp_wp_idx
        return line_wp_idx

    def get_light_state(self):
        """
        determin the current color of the light from image
        Return: int, ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if not self.has_image:
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, desired_encoding="rgb8")
        return self.light_classifier.get_classification(cv_image)

    def update_state(self, light_wp, state):
        '''
        update state and publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def process_traffic_lights_sim(self):
        """
        Finds closest visible traffic light, if one exists, and determines its location and color,
        used only in simulator
        Return:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closestlight, line_wp_idx = self.find_closest_traffic_light_sim()
        if closestlight is None:
            return -1, TrafficLight.UNKNOWN
        light_state = self.get_light_state_sim(closestlight)
        return line_wp_idx, light_state

    def get_light_state_sim(self, light):
        """
        Determines the current color of the traffic light,
        used only in simulator
        Param: light (TrafficLight), light to classify
        Return: int, ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # for testing, just return the light state
        return light.state

    def find_closest_traffic_light_sim(self):
        """
        find the closest traffic light and the stop line waypoint index to current pose,
        used only in simulator
        """
        stop_line_positions = self.config['stop_line_positions']
        closest_light = None
        line_wp_idx = None
        if self.pose:
            car_wp_idx = self.waypoints.get_closest_waypoint_idx(self.pose.pose.position.x, self.pose.pose.position.y)
            # find the closest visible traffic light (if one exists)
            min_distance = self.waypoints.get_num_waypoints()
            for i, light in enumerate(self.lights):
                # get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.waypoints.get_closest_waypoint_idx(line[0], line[1])
                # find closest stop line waypoint index
                distance = temp_wp_idx - car_wp_idx
                if distance >= 0 and distance < min_distance:
                    min_distance = distance
                    closest_light = light
                    line_wp_idx = temp_wp_idx
        return closest_light, line_wp_idx

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
