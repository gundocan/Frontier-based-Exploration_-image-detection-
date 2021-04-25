#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from geometry_msgs.msg import Pose, Pose2D, Quaternion, Transform
import numpy as np
import rospy
from exploration.srv import GenerateFrontier, GenerateFrontierResponse
from exploration.srv import PlanPath, PlanPathRequest, PlanPathResponse
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose2D, Pose, PoseStamped, Point, Quaternion
import tf2_ros
import geometry_msgs.msg
from std_srvs.srv import Trigger, TriggerResponse
from nav_msgs.msg import Path
from exploration.msg import status
from std_msgs.msg import String
import tf.transformations as tft
#from std_msgs.msg import status

def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))


def pose2to3(pose2):
    """Convert Pose2D to Pose."""
    pose3 = Pose()
    pose3.position.x = pose2.x
    pose3.position.y = pose2.y
    rpy = 0.0, 0.0, pose2.theta
    q = quaternion_from_euler(*rpy)
    pose3.orientation = Quaternion(*q)
    return pose3


def tf3to2(tf):
    """Convert Transform to Pose2D."""
    pose2 = Pose2D()
    pose2.x = tf.translation.x
    pose2.y = tf.translation.y
    rpy = euler_from_quaternion(slots(tf.rotation))
    pose2.theta = rpy[2]
    return pose2


def getRobotCoordinates(self):
    """ Get the current robot position in the grid """
    try:
        transfor = self.tfBuffer2.lookup_transform(self.mapFrame, self.robotFrame, rospy.Time(), rospy.Duration(0.5))
        print("Transformation working")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Cannot get the robot position! Explorer")
        self.robotPosition = None
    else:
        x= transfor.transform.translation.x
        y= transfor.transform.translation.y
        print("X :", x)
        print("Y :", y)
        #pos = np.array((transfor.transform.translation.x, transfor.transform.translation.y)).reshape(2,)
        pos = Pose2D(x , y , 0.0)
       
        print("pose : ",pos)
        return pos #transform the robot coordinates from real-world (in meters) into grid






class Explorer(object):
    def __init__(self):

        self.mapFrame = rospy.get_param("~map_frame", "map")
        self.robotFrame = rospy.get_param("~robot_frame", "robot")
        self.robotDiameter = float(rospy.get_param("~robot_diameter", 0.3))
        self.tfBuffer2 = tf2_ros.Buffer()
        listener2 = tf2_ros.TransformListener(self.tfBuffer2)
        self.status_listener = rospy.Subscriber("status_msg", String , queue_size =1 )
        self.path_subscriber = rospy.Subscriber('path', Path, queue_size=1)
        rospy.wait_for_service('get_closest_frontier')
        self.service_closest= rospy.ServiceProxy('get_closest_frontier', GenerateFrontier)
        self.service_random= rospy.ServiceProxy('get_random_frontier', GenerateFrontier)
        self.req1 =self.service_closest()
        


        print("the response is")
        print(self.req1)
        self.resx =(self.req1.goal_pose.x)
        self.resy =(self.req1.goal_pose.y)
        self.restheta =(self.req1.goal_pose.theta)
        self.goal = Pose2D(self.resx , self.resy , self.restheta)
        self.start = getRobotCoordinates(self)
        
       
        rospy.wait_for_service('plan_path')

        self.service_planpath = rospy.ServiceProxy("plan_path",PlanPath)
        
        self.service_clearpath = rospy.ServiceProxy("clear_travelled_path", Trigger )
        
        self.req2 = self.service_planpath(self.start,self.goal)

      
      
        rospy.wait_for_message("status_msg",String)
        self.clear = self.service_clearpath()
        
        rospy.wait_for_service('get_random_frontier')
        self.req3 =self.service_random()
        
        self.resx1 =(self.req3.goal_pose.x)
        self.resy1 =(self.req3.goal_pose.y)
        self.restheta1 =(self.req3.goal_pose.theta)
        self.goal2 = Pose2D(self.resx1 , self.resy1 , self.restheta1)
        self.start2 =  getRobotCoordinates(self)
        
        
        rospy.wait_for_service('plan_path')
        self.req4 = self.service_planpath(self.start2,self.goal2)
      
        
    
        # all is up to you
        pass


if __name__ == '__main__':
    rospy.init_node('explorers', log_level=rospy.INFO)
    
    node = Explorer()
    
    rospy.spin()
