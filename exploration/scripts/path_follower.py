#!/usr/bin/env python
"""
Simple path follower.

Always acts on the last received plan.
An empty plan means no action (stopping the robot).
"""
from __future__ import absolute_import, division, print_function
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, Pose2D, Quaternion, Transform, TransformStamped, Twist, PoseStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
import rospy
from exploration.msg import status
from std_msgs.msg import String
from ros_numpy import msgify, numpify
from scipy.spatial import cKDTree
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_py import TransformException
import tf2_ros
from threading import RLock
from timeit import default_timer as timer
import math
from exploration.utils import getRobotGridPosition, gridToMapCoordinates, mapToGridCoordinates, getGridPosition
np.set_printoptions(precision=3)


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def tf3to2(tf):
    """Converts tf to Pose2D."""
    pose2 = Pose2D()
    pose2.x = tf.translation.x
    pose2.y = tf.translation.y
    rpy = euler_from_quaternion(slots(tf.rotation))
    pose2.theta = rpy[2]
    return pose2


class PathFollower(object):
    def __init__(self):
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom') # No-wait frame
        self.robot_frame = rospy.get_param('~robot_frame', 'robot') # base_footprint for simulation, robot for dummy grid
        self.control_freq = rospy.get_param('~control_freq', 10.0) # control loop frequency (Hz)
        assert 1.0 < self.control_freq < 25.0
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist', 0.2) # allowed distance from goal to be supposed reach  
        self.goal_reached_angle = rospy.get_param('~goal_reached_angle', 0.2) # allowed difference in heading 
        self.use_path_heading = rospy.get_param('~use_path_heading', 'last') # use path heading during planning 
        assert self.use_path_heading in ('none', 'last', 'all')
        self.max_age = rospy.Duration(rospy.get_param('~max_age', 1.0)) # maximum latency of the path message to be considered as valid
        self.max_path_dist = rospy.get_param('~max_path_dist', 4.0) # maximum distance from a path start to enable start of path following
        self.max_velocity = rospy.get_param('~max_velocity', 0.5) # maximumm allowed velocity (m/s)
        self.max_angular_rate = rospy.get_param('~max_angular_rate', 2.0) # maximum allowed angular rate (rad/s)
        self.use_dummy_grid = rospy.get_param('~use_dummy_grid', 1) # set to zero for regular simulation
        self.look_ahead = rospy.get_param('~look_ahead_dist', 0.2) # look ahead distance for pure pursuit (m)
        self.const_velocity = rospy.get_param('~const_velocity', 0.1) # desired constant velocity for path following
        self.const_angular_rate = rospy.get_param('~const_angular_rate', 0.2) #desired constant angular rate for turning on a spot
        self.control_law = rospy.get_param('~control_law', "pure_pursuit") # type of applied control_law approach PID/pure_pursuit/turn_and_move
        # self.control_law = "turn_and_move"
        assert self.control_law in ('PID', 'pure_pursuit', 'turn_and_move', 'simplified_turn_and_move')

        self.lock = RLock()
        self.path_msg = None  # Path message
        self.path = None  # n-by-3 path array
        self.path_x_index = None  # Path position index - KDTree
        self.path_index = 0 # Path position index
        self.status = "I'm done moving"
        self.prev_heading_error = 0.0 # heading error in a previous step
        self.heading_error_int = 0.0 # integrated heading error
        self.status_pub = rospy.Publisher("status_msg", String , queue_size=1 )
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1) # Publisher of velocity commands
        self.lookahead_point_pub = rospy.Publisher('lookahead_point', Marker, queue_size=1) # Publisher of the lookahead point (visualization only)

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        self.path_sub = rospy.Subscriber('path', Path, self.path_received, queue_size=2) # path subscriber
        self.timer = rospy.Timer(rospy.Duration.from_sec(1. / self.control_freq), self.control) # control loop timer

        #  hedefe varildi mi?
        self.goal_reached = 0

        rospy.loginfo('Path follower initialized.')

    def lookup_transform(self, target_frame, source_frame, time,
                         no_wait_frame=None, timeout=rospy.Duration.from_sec(0.0)):
        if no_wait_frame is None or no_wait_frame == target_frame:
            return self.tf.lookup_transform(target_frame, source_frame, time, timeout=timeout)

        tf_n2t = self.tf.lookup_transform(self.map_frame, self.odom_frame, rospy.Time())
        tf_s2n = self.tf.lookup_transform(self.odom_frame, self.robot_frame, time, timeout=timeout)
        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = time
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(Transform,
                                  np.matmul(numpify(tf_n2t.transform),
                                            numpify(tf_s2n.transform)))
        return tf_s2t

    def get_robot_pose(self, target_frame):
        if self.use_dummy_grid:
            tf = self.lookup_transform(target_frame, self.robot_frame, rospy.Time.now(),
                                       timeout=rospy.Duration.from_sec(0.5), no_wait_frame=target_frame)
        else: 
            tf = self.lookup_transform(target_frame, self.robot_frame, rospy.Time.now(),
                                       timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)

        pose = tf3to2(tf.transform)
        return pose

    def clear_path(self):
        self.path_msg = None
        self.path = None
        self.path_x_index = None
        self.path_index = 0

    def path_received(self, msg):
        """Callback method of a path subscriber."""
        age = rospy.Time.now() - msg.header.stamp
        if age > self.max_age:
            rospy.logwarn('Discarding path %.1f s > %.1f s old.', age.to_sec(), self.max_age.to_sec())
            return
        with self.lock:
            if len(msg.poses) > 0:
                self.goal_reached = 0 
                self.path_msg = msg
                self.path = np.array([slots(p.pose.position) for p in msg.poses])
                self.path_x_index = cKDTree(self.path[:, :2]) # [unused]
                self.path_index = 0 # path position index
            else:
                self.clear_path()

        rospy.loginfo('Path received (%i poses).', len(msg.poses))

    def turnRobot(self, angle_to_turn, abs_angular_rate):
        """Sends a command to turn robot by an angle 'angle_to_turn' with an angular rate 'abs_angular_rate'."""
        angle_dir = -1 if angle_to_turn < 0 else 1
        angular_rate = np.clip(abs_angular_rate * angle_dir, -self.max_angular_rate, self.max_angular_rate)
        time_to_turn = angle_to_turn / angular_rate
        msg = Twist()
        msg.angular.z = angular_rate
        msg.linear.x = 0.0
        rospy.loginfo('Turn: velocity: %.2f m/s, angular rate: %.1f rad/s. (%.3f s)',
                      msg.linear.x, msg.angular.z, time_to_turn)
        self.cmd_pub.publish(msg)
        rospy.sleep(time_to_turn)
        self.cmd_pub.publish(Twist())

    def moveRobot(self, dist, velocity):
        """Sends a command to move robot forwards at a distance "dist" with a velocity abs_angular_rate."""
        time_to_move = dist / velocity
        msg = Twist()
        msg.angular.z = 0.0
        msg.linear.x = velocity
        rospy.loginfo('move: velocity: %.2f m/s, angular rate: %.1f rad/s, time to move: (%.3f s)',
                      msg.linear.x, msg.angular.z, time_to_move)
        self.cmd_pub.publish(msg)
        rospy.sleep(time_to_move)
        self.cmd_pub.publish(Twist())

    def simplifyPath(self, original_path):
        """Returns a simplified path that consists of a minimum number of waypoints so that the original path remains unchanged."""
        simplified_path = [] 

        if len(original_path) < 2:
            simplified_path = original_path
            return simplified_path

        simplified_path.append(original_path[0])
        p_dir = (original_path[1] - simplified_path[-1])[:2]
        heading = np.arctan2(p_dir[1], p_dir[0]) # heading towards goal

        for i in np.arange(2, len(original_path)): 
            p_dir = (original_path[i] - simplified_path[-1])[:2]
            new_heading = np.arctan2(p_dir[1], p_dir[0]) # heading towards goal
            if abs(new_heading - heading) > 1e-2 or i == (len(original_path) - 1):
                simplified_path.append(original_path[i-1])
                p_dir = (original_path[i] - simplified_path[-1])[:2]
                heading = np.arctan2(p_dir[1], p_dir[0]) # heading towards goal

        simplified_path.append(original_path[-1])

        return simplified_path

    def turnAndMove(self, path):
        """Performs naive turn and move sequence of commands to navigate along the path."""
        i = 0
        n_waypoints = len(path)
        with self.lock:
            path_msg_timestamp = self.path_msg.header.stamp 

        for goal in path: 
            rospy.loginfo('Turn and move to next goal: [%.2f, %.2f]', goal[0], goal[1])
            i = i + 1
            with self.lock:
                pose_msg = self.get_robot_pose(self.path_msg.header.frame_id) # get robot pose
                pose = np.array(slots(pose_msg))
                if not path_msg_timestamp == self.path_msg.header.stamp:
                    rospy.loginfo('New path message received. Following of current path stopped.')
                    return

            heading = pose[2] # current heading
            p_dir = (goal - pose)[:2] # position displacement (direction)
            goal_heading = np.arctan2(p_dir[1], p_dir[0]) # heading towards goal

            if (self.use_path_heading == 'none' or (self.use_path_heading == 'last' and i < n_waypoints) or np.isnan(goal[2])):
                final_heading = goal_heading
            else:
                final_heading = goal[2]
                rospy.loginfo('Using path heading: %.1f.', final_heading)

            heading = pose[2]

            # Position displacement (Euclidean distance)
            dist = np.linalg.norm(p_dir)

            # Angular displacement from [-pi, pi)
            angle_to_turn = (goal_heading - heading + np.pi) % (2. * np.pi) - np.pi # smaller angle difference

            # Turn
            self.turnRobot(angle_to_turn, self.const_angular_rate)

            # Move
            self.moveRobot(dist, self.const_velocity)

            # Turn to final heading 
            if abs(final_heading - goal_heading) > 1e-5:
                angle_to_turn = (final_heading - goal_heading + np.pi) % (2. * np.pi) - np.pi # smaller angle difference
                self.turnRobot(angle_to_turn, self.const_angular_rate)

            # Stop robot
            self.cmd_pub.publish(Twist())

        rospy.loginfo('Turn and move: last waypoint reached.')
        self.clear_path()
    

    def linePointDist(self, line_begin, line_end, point):
        """Return distance from the line defined by two points (not from the segment)."""
        p = (point - line_begin)
        v = (line_end - line_begin)
        #Burada ABS kullanilirsa donus tek yonlu olur
        # return ((v[0])*(p[1]) - (p[0])*(v[1])) / np.linalg.norm(v[:2])
        return ((v[0])*(p[1]) - (p[0])*(v[1])) / np.linalg.norm(v[:2])

    def setPathIndex(self, pose): 
        # """Set path index indicating path segment lying within lookahed distance."""
        for k in np.arange(self.path_index, len(self.path)):
              
            if np.linalg.norm(self.path[k][:2] - pose[:2]) < self.look_ahead:
                print( "formul : ",np.linalg.norm(self.path[k][:2] - pose[:2])) 
                continue
            else: 
                self.path_index = k 
                print("k2 : ",self.path_index)
                return

        self.path_index = len(self.path) - 1 # all remaining waypoints are close
        print("k son . ", self.path_index)
    
    def getLookaheadPoint(self, pose):
        """Returns lookahead point used as a reference point for path following."""
        
        #lookahead_point =np.array([0.0, 0.0, 0.0])
        self.setPathIndex(pose)
        # durus noktasi ayari, index geride idi
        # p = self.path_index - 1
        p = self.path_index
        #if p == 0 : p==1
        #elif p==-1 : p==0
        print("set Path index", p)
        #print(self.path)
        #print(self.path[2][1])
        #print([self.path[p][0], self.path[p][1], self.path[p][2]])
        lookahead_point = [self.path[p][0], self.path[p][1], self.path[p][2]]


        #TODO: find a lookahead point on a path 

        return lookahead_point 

    def publishGoalPose(self, goal_pose):
        """Publishes goal pose."""
        msg = Marker()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()
        msg.id = 1
        msg.type = 2
        msg.action = 0
        msg.pose = Pose()
        msg.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        msg.pose.position = Point(goal_pose[0], goal_pose[1], 0.0) 
        # m_start.points.append(Point(start.x, start.y, 0.0))
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.scale.x = 0.05
        msg.scale.y = 0.05
        msg.scale.z = 0.01
        self.lookahead_point_pub.publish(msg)

    def sendVelocityCommand(self, linear_velocity, angular_rate):
        """Calls command to set robot velocity and angular rate."""
        msg = Twist()
        
        msg.angular.z = angular_rate
        msg.linear.x = linear_velocity
        self.cmd_pub.publish(msg)

    def control(self, event):
        """Timer callback - main cntrol loop."""
        try:
            t = timer() # save current time
            if self.path_msg is None:
                self.cmd_pub.publish(Twist())
                return

            rospy.loginfo_throttle(1.0, 'Following path.')

            if self.control_law == "turn_and_move":
                self.turnAndMove(self.path)
            elif self.control_law == "simplified_turn_and_move":
                s_path = self.simplifyPath(self.path)
                self.turnAndMove(s_path)
            else:
                with self.lock:
                    if self.path_msg is None:
                        self.cmd_pub.publish(Twist())
                        return
                    
                    # get robot pose
                    #print("slak :",self.path_msg.header.frame_id)
                    pose_msg = self.get_robot_pose(self.path_msg.header.frame_id)
                    #print("pose_msg : ",pose_msg)
                    pose = np.array(slots(pose_msg))
                    #print("Pose : ",pose)
                    #s=self.setPathIndex(pose)
                    #print("always looping")
                    #r=self.setPathIndex(pose)
                    #print("path Index : ",r)
                    
                    #print("z :",z)
                    # get look ahead point
                    goal = self.getLookaheadPoint(pose)
                    self.publishGoalPose(goal)

                  #TODO: Implement desired behaviour if the goal has been reached

                #TODO: Implement desired behaviour if the beginning of path is too far

                angular_rate = 0.0
                velocity = 0.0 

                # set angular rate and velocity based on control law
                if self.control_law == "pure_pursuit": # cannot be combined with goal heading 
                    if (self.goal_reached == 0):
                        dl = math.sqrt( ((pose[0]-goal[0])**2) + ((pose[1]-goal[1])**2) + ((pose[2]-goal[2])**2))
                        #print("dl :", dl)
                        #print("pose-goal :",pose[0]-goal[0])
		                #print("vector :",v[0],v[1])
                        #print("Pose :",pose )
                        #print("goal :",goal)
                        v = [pose[0] + np.cos(pose[2]), pose[1]+np.sin(pose[2]), 0.0]
                        d = self.linePointDist(pose , v, goal)

                        #print("linetoPo :", d)
                        #print("Distance to Look ahead point : ", dl)
                        #print("v vector :", v)
                        radius = (dl**2)/(2*d)
                        #print("radius : ", radius)
                        angular_rate = self.const_velocity / radius
                        # Burada angular_rate i 10 ile carpalim, hizli reaksiyon
                        angular_rate *= 10.0
                        print("Constant velocity :", self.const_velocity)
                        print("Angular velocity calculated :", angular_rate)
                        velocity = self.const_velocity
		                
                        #self.sendVelocityCommand(velocity, angular_rate)
		                
                        #TODO: compute velocity command (linear velocity and angular rate) based on the pure pursuit law

                        rospy.logwarn_throttle(2.0, 'Pure pursuit law is implemented.')

                        # hedefe ulasildi mi bak?
                        if (pose[0] >= goal[0]) and (pose[1] >= goal[1]):
                             angular_rate = 0.0
                             velocity = 0.0
                             self.goal_reached = 1 
                             self.clear_path()
                             status = self.status
                             self.status_pub.publish(status)

                elif self.control_law == "PID":

                    #TODO: compute velocity command (linear velocity and angular rate) based on the PID control law

                    rospy.logwarn_throttle(2.0, 'PID control law not implemented. Sending zero velocity command.')

                # apply limits on angular rate 
                angular_rate = np.clip(angular_rate, -self.max_angular_rate, self.max_angular_rate)

                # apply limits on linear velocity
                velocity = np.clip(velocity, 0., self.max_velocity)

                # Apply desired velocity and angular rate
                self.sendVelocityCommand(velocity, angular_rate)
                rospy.logdebug('Linear velocity: %.2f m/s, angular rate: %.1f rad/s. (%.3f s)',
                               velocity, angular_rate, timer() - t)
                

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)


if __name__ == '__main__':
    rospy.init_node('path_follower', log_level=rospy.INFO)
    node = PathFollower()
    rospy.spin()
