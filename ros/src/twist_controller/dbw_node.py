#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        self.controller = Controller(
            vehicle_mass=vehicle_mass,
            fuel_capacity=fuel_capacity,
            brake_deadband=brake_deadband,
            decel_limit=decel_limit,
            accel_limit=accel_limit,
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)

        # init the member variables to None
        self.throttle = 0
        self.brake = 0
        self.steering = 0

        self.curr_ang_vel = None

        self.current_vel = None
        self.linear_vel = None
        self.angular_vel = None
        self.dbw_enabled = None
        self.final_waypoints = None

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('final_waypoints', Lane, self.final_waypoints_cb)

        self.loop()

    def loop(self):
        # dbw requires the controls to be >= 50 Hz
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if not None in (self.current_vel, self.linear_vel, self.angular_vel, self.final_waypoints):
                self.cte = self.calc_cte(self.final_waypoints, self.pose)
                self.throttle, self.brake, self.steering = self.controller.control(
                    self.dbw_enabled,
                    self.current_vel,
                    self.linear_vel,
                    self.angular_vel,
                    self.cte)
            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steering)
            rate.sleep()

    def dbw_enabled_cb(self, dbw_enabled):
        self.dbw_enabled = dbw_enabled

    def twist_cb(self, twist_msg):
        self.linear_vel = twist_msg.twist.linear.x
        self.angular_vel = twist_msg.twist.angular.z

    def velocity_cb(self, velocity_msg):
        self.current_vel = velocity_msg.twist.linear.x

    def pose_cb(self, pose):
        self.pose = pose

    def final_waypoints_cb(self, waypoints):
        self.final_waypoints = waypoints.waypoints

    def get_xy(self,wp):
        coords = []

        for n in range(len(wp)):
            coords.append([wp[n].pose.pose.position.x, wp[n].pose.pose.position.y])

        return coords

    def calc_cte(self,waypoints,pose):
        origin = waypoints[0].pose.pose.position

        wps = self.get_xy(waypoints)

        relative_wps = wps - np.array([origin.x, origin.y])
        relative_pose = np.array([pose.pose.position.x - origin.x, pose.pose.position.y - origin.y])

        psi = np.arctan2(relative_wps[10,1], relative_wps[10,0])

        rotate = np.array([[np.cos(psi), -np.sin(psi)],
                          [np.sin(psi), np.cos(psi)]])

        rotated_wps = np.dot(relative_wps, rotate)
        rotated_pose = np.dot(relative_pose, rotate)

        coeffs = np.polyfit(rotated_wps[:, 0], rotated_wps[:, 1], 2)

        cte = np.polyval(coeffs,rotated_pose[0]) - rotated_pose[1]

        return cte

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
