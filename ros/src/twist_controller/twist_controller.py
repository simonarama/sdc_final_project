import rospy

from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband,
        decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio,
        max_lat_accel, max_steer_angle):

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle

        # init yaw controller
        min_speed = 0.1
        self.yaw_controller = YawController(
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            min_speed=min_speed,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)

        # init throttle controller

        # kp,ki,kd 
        t_vals = [0.3,0.1,0.0]
        # s_vals = [0.2,0.0001,0.5]
        s_vals = [0.15,0.001,0.1]

        min_throttle = 0.0
        max_throttle = 0.35

        self.throttle_controller = PID(kp=t_vals[0], ki=t_vals[1], kd=t_vals[2], mn=min_throttle, mx=max_throttle)
        self.steering_controller = PID(kp=s_vals[0], ki=s_vals[1], kd=s_vals[2], mn=-self.max_steer_angle, mx=self.max_steer_angle)

        # init velocity low pass filter, use low pass filter to filter out high frequency noise in velocity
        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        # tau = 0.85 # 1/(2pi*tau) = cutoff frequency
        # ts = 0.02 # sample time
        # self.str_lpf = LowPassFilter(tau, ts)

        # setup last time value
        self.last_time = rospy.get_time()

    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel, cte):
        # first check if the dbw is enabled or not
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.steering_controller.reset()
            return 0.0, 0.0, 0.0

        # get the current
        current_vel = self.vel_lpf.filt(current_vel)
        # cte = self.str_lpf.filt(cte)

        yc_steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        # print "CTE:", cte

        throttle = self.throttle_controller.step(vel_error, sample_time)
        pid_steering = self.steering_controller.step(cte, sample_time)

        # steering = pid_steering
        # steering = yc_steering
        steering = yc_steering + pid_steering
        # steering = (yc_steering + pid_steering) / 2

        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0
            brake = 400
            self.throttle_controller.reset()
            self.steering_controller.reset()
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius

        return throttle, brake, steering
