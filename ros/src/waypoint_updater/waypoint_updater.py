#!/usr/bin/env python

import math
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

from common.waypoints import WayPoints

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        # TODO: Add other member variables you need below
        self.pose = None
        self.waypoints = WayPoints()

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        # set the loop rate to be 50
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints.has_waypoints():
                x = self.pose.pose.position.x
                y = self.pose.pose.position.y
                closest_waypoint_idx = self.waypoints.get_closest_waypoint_idx(x, y)
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def publish_waypoints(self, closest_waypoint_index):
        lane = self.waypoints.get_lane_with_waypoints(closest_waypoint_index, closest_waypoint_index + LOOKAHEAD_WPS)
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, pose):
        self.pose = pose

    def waypoints_cb(self, waypoints):
        self.waypoints.set_waypoints(waypoints)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
