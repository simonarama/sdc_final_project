import numpy as np

from scipy.spatial import KDTree
from styx_msgs.msg import Lane, Waypoint


class WayPoints:
    def __init__(self):
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None

    def has_waypoints(self):
        return self.base_waypoints is not None

    def set_waypoints(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def get_num_waypoints(self):
        return len(self.base_waypoints.waypoints)

    def get_closest_waypoint_idx(self, x, y):
        """get the index of closest waypoint ahead of position(x, y)"""
        # get the index of the closest waypoint of current pose
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # get the closest waypoint coord using the index
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # check if the closest waypoint is in front of the car or behind it,
        # if behind it, use the next way point
        # equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def get_lane_with_waypoints(self, start_wp_idx, end_wp_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[start_wp_idx:end_wp_idx]
        return lane