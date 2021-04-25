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
from queue import PriorityQueue

"""
Here are imports that you are most likely will need. However, you may wish to remove or add your own import.
"""


class PathPlanner():

    def __init__(self):
        # Initialize the node
        rospy.init_node("path_planner")
        self.res = 0
        # Get some useful parameters
        self.mapFrame = rospy.get_param("~map_frame", "map")
        self.robotFrame = rospy.get_param("~robot_frame", "base_footprint")
        self.robotDiameter = float(rospy.get_param("~robot_diameter", 0.3))
        self.occupancyThreshold = int(
            rospy.get_param("~occupancy_threshold", 10))
        self.grid = []
        self.gridInfo = 0
        # Helper variable to determine if grid was received at least once
        self.gridReady = False
        self.lookAroundSteps = 0
        # You may wish to listen to the transformations of the robot
        self.tfBuffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        listener = tf2_ros.TransformListener(self.tfBuffer)
        self.gridUpdated = False
        self.threshold = 0
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
        # m_start.points.append(Point(start.x, start.y, 0.0))
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
        # m_start.points.append(Point(start.x, start.y, 0.0))
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

        startPosition = getGridPosition(request.start, self.gridInfo)
        print("start pos: ", startPosition)
        goalPosition = getGridPosition(request.goal, self.gridInfo)
        print("goal pos: ", goalPosition)

        self.publishStartAndGoal(request.start, request.goal)

        # TODO:
        # TODO: First, you should try to obtain the robots coordinates

        # TODO: Then, copy the occupancy grid into some temporary variable and inflate the obstacles

        # TODO: Compute the path, i.e. run some graph-based search algorithm
        path = []
        path = self.aStar(startPosition, goalPosition)
        print("A star is done .")
        #print("path :", path)
        response = PlanPathResponse()

        real_path = [Pose2D(pos[0], pos[1], 0) for pos in
                     [gridToMapCoordinates(waypoint, self.gridInfo) for waypoint in path]]
        response = PlanPathResponse(real_path)

        # Publish planned path for visualization
        self.publishPath(response.path)

        return response

    def aStar(self, start, goal):
        visited = []
        goal = [int(goal[0]), int(goal[1])]
        start = [int(start[0]), int(start[1])]
        frontier = PriorityQueue()
        path = []
        bestPathSoFar = [9999999, []]
        found = False
        mult = 75
        print(goal)

        H = self.heuristic(start, goal)
        frontier.put((H, 0, start, [start]))

       # print("H:", H, "*", mult, "=", H*mult)

        dist = [[1, 0], [-1, 0], [0, 1], [0, -1],
                [1, 1], [-1, 1], [1, -1], [-1, -1]]
        # print(self.width*self.height)
        # print(self.grid[start[0]][start[1]])
        # print(self.grid[goal[0]][goal[1]])
        H = H * mult
        while not frontier.empty():

            current = frontier.get()
            # print("New iter #")
            #print("Current :" ,current)
            # print(frontier.qsize())
            #print(current[2], goal)
            if current[2] == goal:
                path = current[3]
                break
            if (current[2] in visited):
                continue
            else:
                visited.append(current[2])

            if (len(visited) > H):
                break

            for k in range(len(dist)):
                try:
                    #print("Cons: ", current[2][0] + dist[k][0], current[2][1] + dist[k][1])
                    if ([current[2][0] + dist[k][0], current[2][1] + dist[k][1]] == goal):
                        print("GOAL IS FOUND!")
                        found = True
                        cost = current[1] + 1
                        if (k >= 4):
                            cost += 0.41

                        path_cp = current[3][:]
                        path_cp.append(goal)
                        frontier = PriorityQueue()
                        frontier.put((priority, cost, goal, path_cp))
                    else:
                       # print("you are in else ")
                        #print(current[2][0] + dist[k][0], current[2][1] + dist[k][1], dist[k], self.grid[current[2][0] + dist[k][0]][current[2][1] + dist[k][1]])
                        if (not found and ((self.grid[current[2][0] + dist[k][0]][current[2][1] + dist[k][1]] < 26 and self.grid[current[2][0] + dist[k][0]][current[2][1] + dist[k][1]] >= 0) or len(current[3]) < self.threshold)):
                            ind = [current[2][0] + dist[k][0],
                                   current[2][1] + dist[k][1]]

                            if (ind not in visited):

                                cost = current[1] + 1
                                if (k >= 4):
                                    cost += 0.41
                                # print("HERE")
                                heur = self.heuristic(ind, goal)
                                priority = cost + heur
                                path_cp = current[3][:]

                                path_cp.append(ind)
                                frontier.put((priority, cost, ind, path_cp))
                                #print("ADDED: ", priority, cost, ind, path_cp)
                                if bestPathSoFar[0] > heur:
                                    bestPathSoFar = [heur, path_cp]
                                # visited.append(ind)

                except:
                    pass

        path_fin = []

        if (len(path) == 0):
            print("CANNOT FIND A PATH!")
            print(bestPathSoFar)
            # return bestPathSoFar[1]
            if (bestPathSoFar[0] < 35 and len(bestPathSoFar[1]) > 12):
                path = bestPathSoFar[1]
        # if (len(path) <= 10 or len(path)>75): path =
        print("path : ", path)
        for ind2, n in enumerate(path):
            #    if (self.indToGrid(n)):
            # print(self.grid[n[0]][n[1]])
           # print(ind2, n , self.grid[n[0]][n[1]])
            #path_fin.append(np.array((n[0],n[1]), dtype=float))
            if (ind2 < (self.threshold) or (self.grid[n[0]][n[1]] <= 26 and self.grid[n[0]][n[1]] >= 0)):
                path_fin.append(np.array((int(n[0]), int(n[1])), dtype=int))

        return path_fin

    def getRobotCoordinates(self):
        """ Get the current robot position in the grid """
        try:
            trans = self.tfBuffer.lookup_transform(
                self.mapFrame, self.robotFrame, rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Cannot get the robot position Planner!")
            self.robotPosition = None
        else:
            pos = np.array((trans.transform.translation.x, trans.transform.translation.y)).reshape(
                2, )  # TODO: transform the robot coordinates from real-world (in meters) into grid

            gridPos = getRobotGridPosition(pos, self.gridInfo)
            return gridPos

    def heuristic(self, pos1, pos2):
        # manhattan distance
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def extractGrid(self, msg):
        # TODO: extract grid from msg.data and other usefull information
        self.res = msg.info.resolution
        self.width = msg.info.width
        self.gridInfo = msg.info
        # print(self.gridInfo)
        self.height = msg.info.height
        self.origin = msg.info.origin
        self.position = msg.info.origin.position
        self.orientation = msg.info.origin.orientation
        if (self.lookAroundSteps == 0):
            self.lookAroundSteps = 2 * \
                float(self.robotDiameter) / float(self.res)
            self.lookAroundSteps = int(round(self.lookAroundSteps, 3))
           # print("lookout distnce ", self.lookAroundSteps)
            tmp = round(self.lookAroundSteps)
            if (tmp >= self.lookAroundSteps):
                self.lookAroundSteps = int(tmp)
            else:
                self.lookAroundSteps = int(tmp) + 1
            self.lookAroundStepsTmp = self.lookAroundSteps
        self.grid = np.reshape(msg.data, (self.height, self.width)).T
        # print(self.grid)
        # aa=self.grid[8][7]
        #print("A :", aa)
        # for j in range(self.height):
        # str = ""
        # for i in range(self.width):
        #  if (self.grid[i][j] >0): str += "#"
        #  else: str += " "
        # print(str)
        print("UPDATED GRID MAP")
        self.grid = (morphology.grey_dilation(self.grid, size=(3, 3)))
        # for j in range(self.height):
        # str = ""
        # for i in range(self.width):
        # if (self.grid[i][j] >0): str += "#"
        # else: str += " "
        # print(str)
        self.threshold = 3
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
