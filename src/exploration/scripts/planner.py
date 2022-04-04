#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import rospy
import numpy as np
from scipy.ndimage import morphology
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose2D, Pose, PoseStamped, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from exploration.srv import PlanPath, PlanPathRequest, PlanPathResponse
from exploration.utils import getRobotGridPosition, gridToMapCoordinates, mapToGridCoordinates, getGridPosition
import tf2_ros
import geometry_msgs.msg
from Queue import PriorityQueue

"""
Here are imports that you are most likely will need. However, you may wish to remove or add your own import.
"""



class PathPlanner():

    def __init__(self):
        # Initialize the node
        rospy.init_node("path_planner")
        self.resolution = 0
        # Get some useful parameters
        self.mapFrame = rospy.get_param("~map_frame", "map")
        self.robotFrame = rospy.get_param("~robot_frame", "base_footprint")
        self.robotDiameter = float(rospy.get_param("~robot_diameter", 0.3))
        self.occupancyThreshold = int(rospy.get_param("~occupancy_threshold", 10))
        self.grid = []
        self.gridInfo = 0
        # Helper variable to determine if grid was received at least once
        self.gridReady = False
        # You may wish to listen to the transformations of the robot
        self.tfBuffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        listener = tf2_ros.TransformListener(self.tfBuffer)
        self.gridUpdated = False
        self.variable = 0
        self.Updated = True

        # Subscribe to grid
        self.gridSubscriber = rospy.Subscriber(
            'occupancy', OccupancyGrid, self.grid_cb)

        # Publishers for visualization
        self.path_vis_pub = rospy.Publisher('path', Path, queue_size=1)
        self.start_and_goal_vis_pub = rospy.Publisher(
            'start_and_goal', MarkerArray, queue_size=1)

        rospy.loginfo('Path planner initialized.')

    def publishPath(self, path_2d):
        msg = Path()
        msg.header.frame_id = self.mapFrame
        msg.header.stamp = rospy.get_rostime()
        for waypoint in path_2d:
            pose = PoseStamped()
            pose.header.frame_id = self.mapFrame
            pose.pose.position.x = waypoint.x
            pose.pose.position.y = waypoint.y
            pose.pose.position.z = 0
            msg.poses.append(pose)

        rospy.loginfo("Publishing plan.")
        self.path_vis_pub.publish(msg)

    def publishStartAndGoal(self, start, goal):

        # Helpful visualization
        msg = MarkerArray()
        m_start = Marker()
        m_start.header.frame_id = self.mapFrame
        m_start.id = 1
        m_start.type = 2
        m_start.action = 0
        m_start.pose = Pose()
        m_start.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        m_start.pose.position = Point(start.x, start.y, 0.0)
        m_start.color.r = 1.0
        m_start.color.g = 0.0
        m_start.color.b = 0.0
        m_start.color.a = 0.8
        m_start.scale.x = 0.1
        m_start.scale.y = 0.1
        m_start.scale.z = 0.001
        msg.markers.append(m_start)

        # goal marker
        m_goal = Marker()
        m_goal.header.frame_id = self.mapFrame
        m_goal.id = 2
        m_goal.type = 2
        m_goal.action = 0
        m_goal.pose = Pose()
        m_goal.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        m_goal.pose.position = Point(goal.x, goal.y, 0.0)
        m_goal.color.r = 0.0
        m_goal.color.g = 1.0
        m_goal.color.b = 0.0
        m_goal.color.a = 0.8
        m_goal.scale.x = 0.1
        m_goal.scale.y = 0.1
        m_goal.scale.z = 0.001
        msg.markers.append(m_goal)
        rospy.loginfo("Publishing start and goal markers.")
        self.start_and_goal_vis_pub.publish(msg)

    def planPath(self, request):
        """ Plan and return path from the robot position to the requested goal """
        # Get the position of the goal (real-world)

        self.Updated = True
        while (not self.gridUpdated):
            pass
        self.gridUpdated = False

        start_position = getGridPosition(request.start, self.gridInfo)
        print("start pos: ", start_position)
        goal_position = getGridPosition(request.goal, self.gridInfo)
        print("goal pos: ", goal_position)

        self.publishStartAndGoal(request.start, request.goal)

        # TODO:
        # TODO: First, you should try to obtain the robots coordinates

        # TODO: Then, copy the occupancy grid into some temporary variable and inflate the obstacles

        # TODO: Compute the path, i.e. run some graph-based search algorithm
        path = []
        path = self.A_star(start_position, goal_position)
        response = PlanPathResponse()
        actual_path = [Pose2D(pos[0], pos[1], 0) for pos in
                     [gridToMapCoordinates(waypoint, self.gridInfo) for waypoint in path]]
        response = PlanPathResponse(actual_path)

        # Publish planned path for visualization
        self.publishPath(response.path)

        return response

    def A_star(self, start, dest):

        dest = [int(dest[0]), int(dest[1])]
        start = [int(start[0]), int(start[1])]
        frontier = PriorityQueue()
        path = []
        path_best = [9999999, []]
        found = False
        arrived = []
        costM = 75
        print(dest)

        H = self.heur(start, dest)
        frontier.put((H, 0, start, [start]))



        close_distance = [[1, 0], [-1, 0], [0, 1], [0, -1],
                [1, 1], [-1, 1], [1, -1], [-1, -1]]

        H = H * costM
        while not frontier.empty():

            curr = frontier.get()

            if curr[2] == dest:
                path = curr[3]
                break
            if (curr[2] in arrived):
                continue
            else:
                arrived.append(curr[2])

            if (len(arrived) > H):
                break

            for i in range(len(close_distance)):
                try:

                    if ([curr[2][0] + close_distance[i][0], curr[2][1] + close_distance[i][1]] == dest):
                        print("GOAL IS FOUND!")
                        found = True
                        cost = curr[1] + 1
                        if (i >= 4):
                            cost += 0.41

                        path_var = curr[3][:]
                        path_var.append(dest)
                        frontier = PriorityQueue()
                        frontier.put((sequence, cost, dest, path_var))
                    else:

                        if (not found and ((self.grid[curr[2][0] + close_distance[i][0]][curr[2][1] + close_distance[i][1]] < 26 and self.grid[curr[2][0] + close_distance[i][0]][curr[2][1] + close_distance[i][1]] >= 0) or len(curr[3]) < self.variable)):
                            index = [curr[2][0] + close_distance[i][0],
                                   curr[2][1] + close_distance[i][1]]

                            if (index not in arrived):

                                cost = curr[1] + 1
                                if (i >= 4):
                                    cost += 0.41

                                heur = self.heur(index, dest)
                                sequence = cost + heur
                                path_var = curr[3][:]

                                path_var.append(index)
                                frontier.put((sequence, cost, index, path_var))

                                if path_best[0] > heur:
                                    path_best = [heur, path_var]


                except:
                    pass

        path_list = []

        if (len(path) == 0):
            print("CANNOT FIND A PATH!")
        path =path_best[1] 

            #if (path_best[0] < 35 and len(path_best[1]) > 12):
             #   path = path_best[1]
        # if (len(path) <= 10 or len(path)>75): path =
        print("path : ", path)
        for index2, p in enumerate(path):
            #path_list.append(np.array((p[0],p[1]), dtype=float))
            #if (index2 < (self.variable) or (self.grid[p[0]][p[1]] <= 26 and self.grid[p[0]][p[1]] >= 0)):
            path_list.append(np.array((int(p[0]), int(p[1])), dtype=int))
        print("path last :",path_list)
        return path_list

    def getRobotCoordinates(self):
        """ Get the current robot position in the grid """
        try:
            trans = self.tfBuffer.lookup_transform(
                self.mapFrame, self.robotFrame, rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Cannot get the robot position Planner!")
            self.robotPosition = None
        else:
            position = np.array((trans.transform.translation.x, trans.transform.translation.y)).reshape(
                2, )  # TODO: transform the robot coordinates from real-world (in meters) into grid

            grid_position = getRobotGridPosition(position, self.gridInfo)
            return grid_position

    def heur(self, pos1, pos2):
        # manhattan distance
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def extractGrid(self, msg):
        # TODO: extract grid from msg.data and other usefull information
        self.resolution = msg.info.resolution
        self.width = msg.info.width
        self.gridInfo = msg.info

        self.height = msg.info.height
        self.origin = msg.info.origin
        self.position = msg.info.origin.position
        self.orientation = msg.info.origin.orientation
        self.grid = np.reshape(msg.data, (self.height, self.width)).T
        print("UPDATED GRID MAP")
        self.grid = (morphology.grey_dilation(self.grid, size=(20, 20)))
        self.variable = 3
        self.Updated = False
        self.gridUpdated = True

        pass

    def grid_cb(self, msg):
        if (self.Updated):
            self.extractGrid(msg)
        if not self.gridReady:
            # TODO: Do some initialization of necessary variables

            # Create services

            self.plan_service = rospy.Service(
                'plan_path', PlanPath, self.planPath)
            self.gridReady = True


if __name__ == "__main__":
    pp = PathPlanner()

    rospy.spin() 
