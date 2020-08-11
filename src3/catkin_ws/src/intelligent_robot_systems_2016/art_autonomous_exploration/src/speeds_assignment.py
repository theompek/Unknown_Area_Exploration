#!/usr/bin/env python

import rospy
import math
import cmath
import time
import numpy as np

from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from sonar_data_aggregator import SonarDataAggregator
from laser_data_aggregator import LaserDataAggregator
from navigation import Navigation


# Class for assigning the robot speeds
class RobotController:

    # Constructor
    def __init__(self):

        # Debugging purposes
        self.print_velocities = rospy.get_param('print_velocities')

        # Where and when should you use this?
        self.stop_robot = False

        # Create the needed objects
        self.sonar_aggregation = SonarDataAggregator()
        self.laser_aggregation = LaserDataAggregator()
        self.navigation = Navigation()

        self.linear_velocity = 0
        self.angular_velocity = 0
        self.localMinimumCount = 0
        self.previous_l_temp = 0
        self.previous_a_temp = 0
        self.diff_a_temp = 0
        self.oscillations_count = 0
        self.stack_count = 0
        self.too_slow_count = 0
        self.max_vel_count = 0

        # Check if the robot moves with target or just wanders
        self.move_with_target = rospy.get_param("calculate_target")

        # The timer produces events for sending the speeds every 110 ms
        rospy.Timer(rospy.Duration(0.11), self.publishSpeeds)
        self.velocity_publisher = rospy.Publisher( \
            rospy.get_param('speeds_pub_topic'), Twist, \
            queue_size=10)

    # This function publishes the speeds and moves the robot
    def publishSpeeds(self, event):

        # Produce speeds
        self.produceSpeeds()

        # Create the commands message
        twist = Twist()
        twist.linear.x = self.linear_velocity
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.angular_velocity

        # Send the command
        self.velocity_publisher.publish(twist)

        # Print the speeds for debuggind purposes
        if self.print_velocities == True:
            print "[L,R] = [" + str(twist.linear.x) + " , " + \
                  str(twist.angular.z) + "]"

    # Produces speeds from the laser
    def produceSpeedsLaser(self):
        ############################### NOTE QUESTION ############################
        # Check what laser_scan contains and create linear and angular speeds
        # for obstacle avoidance
        scan = self.laser_aggregation.laser_scan
        linear = 0
        angular = 0
        max_out_rad = 0.6
        min_in_rad = 0.3
        ang_acc = 5
        angle1 = 1.2
        angle2 = 1.7
        r = 0
        phi = 0
        phi_increment = 0.00628
        phi_min = -2.094
        obstacle_vector = cmath.rect(r, phi)  # Vector in polar format
        normalizer = 0.00001

        # We calculate a vector toward to object(s)
        for index in range(0, len(scan) - 1):
            distance = scan[index]
            r = (max_out_rad - distance) / (max_out_rad - min_in_rad) if max_out_rad > distance else 0
            if r != 0:
                normalizer += 1

            if r > 1:
                r = 1000  # We set a big value if obstacle too close

            phi = phi_min + index * phi_increment
            #r = r * math.exp(-abs(phi)**2)
            obstacle_vector += cmath.rect(r, phi)

        # Normalazation
        obstacle_vector = obstacle_vector / normalizer
        [r, phi] = cmath.polar(obstacle_vector)

        r = r if r < 1 else 1

        const_a = np.sign(-phi) / (1 + abs(phi)) if phi != 0 else 1
        const_b = r if abs(phi) < angle1 else 0
        const_c = r if abs(phi) < angle2 else 0
        const_d = 1 - (math.pi - abs(phi)) / math.pi  # When phi->0 then ang_d-->0 and when phi--> pi then ang_d-->1

        angular = (const_a + np.sign(const_a) * const_b * 5) / 6 if const_b != 0 else ang_acc * const_a * const_c
        linear = const_c*(const_d - 1)  # If an obstacle is too close then negative velocity

        angular = angular if abs(angular) < 1 else np.sign(angular)

        if r == 1 and abs(phi) < 0.2:
            linear = 0

        phi = -math.pi if phi == 0 and r == 0 else phi
        ##########################################################################
        return [linear, angular]

    # Combines the speeds into one output using a motor schema approach
    def produceSpeeds(self):

        # Produce target if not existent
        if self.move_with_target == True and \
                self.navigation.target_exists == False:
            # Creat`    e the commands message
            twist = Twist()
            twist.linear.x = 0
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0

            # Send the command
            self.velocity_publisher.publish(twist)
            self.navigation.selectTarget()

        # Get the submodule's speeds
        [l_laser, a_laser] = self.produceSpeedsLaser()

        # You must fill these
        self.linear_velocity = 0
        self.angular_velocity = 0

        if self.move_with_target == True:
            [l_goal, a_goal] = self.navigation.velocitiesToNextSubtarget()
            ############################### NOTE QUESTION ############################
            # You must combine the two sets of speeds. You can use motor schema,
            # subsumption of whatever suits your better.

            # The tho sets of speeds,
            l_temp = l_goal + 0.4*l_laser**9
            a_temp = a_goal + 0.4*a_laser**15

            # If we get into a local minimum then take control to unstack the vehicle
            l_temp, a_temp = self.local_minimum_unstack(l_temp, a_temp, l_laser, a_laser, a_goal)

            # Speeds into [-0.3 , 0.3]
            self.linear_velocity = 0.3 * np.sign(l_temp) if abs(l_temp) > 0.3 else l_temp
            self.angular_velocity = 0.3 * np.sign(a_temp) if abs(a_temp) > 0.3 else a_temp
            ##########################################################################
        else:
            ############################### NOTE QUESTION ############################
            # Implement obstacle avoidance here using the laser speeds.
            # Hint: Subtract them from something constant
            l_temp = (0.3 + 0.7*l_laser)
            a_temp = 0.3 * a_laser
            self.linear_velocity = 0.3 * np.sign(l_temp) if abs(l_temp) > 0.3 else l_temp
            self.angular_velocity = 0.3 * np.sign(a_temp) if abs(a_temp) > 0.3 else a_temp
            ##########################################################################

    # Assistive functions
    def stopRobot(self):
        self.stop_robot = True

    def resumeRobot(self):
        self.stop_robot = False

    def local_minimum_unstack(self, l_temp, a_temp, l_laser, a_laser, a_goal):
        # -----We take into consideration some cases with local minimum deadlock----
        # If we stack front of an obstacle
        stack_count = 20
        if l_temp == self.previous_l_temp and a_temp == self.previous_a_temp and l_temp < 0.25:
            self.stack_count += 1
        else:
            self.stack_count = 0

        # Or if vehicle oscillates between two positions
        osc_count = 3
        if self.previous_a_temp - a_temp == -self.diff_a_temp and self.diff_a_temp != 0 and l_temp < 0.1:
            self.oscillations_count += 1

        # Or if vehicle moves too slow
        too_slow_count = 1
        if abs(l_temp) < 0.01 and abs(a_temp) < 0.01:
            self.too_slow_count += 1
        else:
            self.too_slow_count = 0

        # Special cases correction
        if self.localMinimumCount != 0 and l_laser == 0 and a_laser == 0 and self.oscillations_count < osc_count:
            self.localMinimumCount = 0

        # If any of the above cases is true then we are in a local minimum
        # and we just use obstacle avoidance velocities
        if self.too_slow_count == too_slow_count or self.stack_count == stack_count or \
                self.oscillations_count > osc_count or self.localMinimumCount != 0:
            l_temp = (0.3 + 0.7 * l_laser)
            a_temp = (-np.sign(a_goal)*(0.3-abs(a_goal))/(0.3+abs(a_goal)))**15 + 0.4 * a_laser
            self.localMinimumCount += 1
            print "Local Minimum"
            if self.too_slow_count == too_slow_count:
                print "Too slow velocities"
            elif self.stack_count == stack_count:
                print "Stack front of an obstacle"
            elif self.oscillations_count > osc_count:
                print "Oscillates near an obstacle"

        # If we will get over the local minimum then we continue
        # for some iterations and then we give back the control
        if l_temp > 0.28:
            self.max_vel_count += 1
        else:
            self.max_vel_count = 0

        # We count for a while until robot get over the local minimum
        upper_limit = 120
        if self.localMinimumCount == upper_limit or self.max_vel_count == 3 * osc_count:
            self.localMinimumCount = 0
            self.oscillations_count = 0
            self.stack_count = 0
            self.max_vel_count = 0

        # We save data regarding the previous velocities
        self.diff_a_temp = self.previous_a_temp - a_temp if self.previous_a_temp - a_temp != 0 else self.diff_a_temp
        self.previous_l_temp = l_temp
        self.previous_a_temp = a_temp

        return l_temp, a_temp
        # ------Local minimum end---------


