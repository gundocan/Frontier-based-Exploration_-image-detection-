#!/usr/bin/env python
from __future__ import division, print_function
import rospy
import numpy as np
from scipy.ndimage import morphology
from nav_msgs.msg import OccupancyGrid, MapMetaData 
from geometry_msgs.msg import Pose2D
from exploration.srv import GenerateFrontier, GenerateFrontierResponse
import tf2_ros
import geometry_msgs.msg
from Queue import *
import tf2_ros
from collections import Counter
import math
#from exploration.msg import sendOccupancy
from geometry_msgs.msg import TransformStamped, Transform
from geometry_msgs.msg import Vector3
import tf.transformations as tft
"""
Here are imports that you are most likely will need. However, you may wish to remove or add your own import.
"""


class FrontierExplorer():

    def __init__(self):
        # Initialize the node
        rospy.init_node("frontier_explorer")
        self.res = 0
        self.var = 0
        self.grid = []
        self.grid2 = []
        self.definetly_Updated= False
        self.Updated = True
        self.gridInfo = []
        self.frontiersList = []
        # Get some useful parameters
        self.mapFrame = rospy.get_param("~map_frame", "map")
        self.robotFrame = rospy.get_param("~robot_frame", "base_footprint")
        self.robotDiameter = float(rospy.get_param("~robot_diameter", 0.2))
        self.occupancyThreshold = int(rospy.get_param("~occupancy_threshold", 10))
        self.gridReady = False
        # You may wish to listen to the transformations of the robot
        self.tfBuffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.matrix_R= np.empty((2, 2))
        self.matrix_t = np.empty((2,))
        # Subscribe to grid
        self.gridSubscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)
        rospy.loginfo('Frontier initialized.')
        # TODO: you may wish to do initialization of other variables
    
    def WFD(self,robotInd):
       # print("wfd algorithm")
        qM = Queue()
        mol = [] #create lists
        mcl = []
        fol = []
        fcl = []
        stat = {}
        for i in range (0, self.width * self.height):
             stat[i] = "none"
        frontiers = []
        qM.put(robotInd)
        stat[robotInd] = "mol"
        dist = [1, self.width, -1, -self.width, self.width - 1, self.width + 1, -self.width + 1, -self.width - 1]
        # check neigbors
        while not qM.empty() :
            p=qM.get()

            if (stat[p] == "mcl"): continue

            if self.transformation_IndexToGrid(p) and self.is_it_frontier(p):
                qF = Queue()
                nf = []
                qF.put(p)
                stat[p] = "fol"

                while not qF.empty() :
                    q =qF.get()
                    if(stat[q] == "mol" or stat[q] == "fcl"): continue
                    if (self.transformation_IndexToGrid(q) and self.is_it_frontier(q)):
                        nf.append(q)
                        for k in dist :
                            if (self.transformation_IndexToGrid(q + k) and stat[q + k] != "fol" and stat[q + k] != "fcl" and stat[q + k] != "mcl"):
                                qF.put(q+k)
                                stat[q + k] = "fol"

                    stat[q] = "fcl"

                if len(nf) >= 1:
                    for i in nf:

                        frontiers.append(i)
                        stat[i] = "mcl"

            for v in dist :
                if(self.transformation_IndexToGrid(p + v)) :
                    if (stat[v + p]!= "mol" and stat[v + p] != "mcl") :
                        for x in dist:
                            if(self.transformation_IndexToGrid(v + p + x) and (self.grid[p + v + x] == 0)) :
                                qM.put((p+v))
                                stat[p + v]= "mol"
                                break
            stat[p]= "mcl"
        #rospy.logwarn("Print Frontiers",frontiers)
        return np.random.choice(frontiers,4)
        #np.random.choice(frontiers, 5)







    def computeWFD(self):
        """ Run the Wavefront detector """

        frontiers = []
        # TODO: First, you should try to obtain the robots coordinates
        # TODO: Then, copy the occupancy grid into some temporary variable and inflate the obstacles
        robot_pose = self.getRobotCoordinates()
        robot_index = 0
        if (self.transformation_GridToIndex(robot_pose)): robot_index = self.var

        frontiers = self.WFD(robot_index)
        print("Frontiers --> ",frontiers) # TODO: Run the WFD algorithm - see the presentation slides for details on how to implement it
        
        return frontiers



    def is_it_frontier(self, index):
        if (self.grid[index] != -1): return False
        if (index % self.width == 0): close_distance = [1, self.width, self.width + 1, -self.width, -self.width + 1]
        elif (index % self.width == self.width-1): close_distance = [-1, self.width, self.width - 1, -self.width, -self.width - 1]
        else: close_distance = [1, self.width, -1, -self.width, self.width - 1, self.width + 1, -self.width + 1, -self.width - 1]
        for x in close_distance:
            if self.transformation_IndexToGrid(index + x):
                if self.grid[index+x] == 0: return True
        return False


    def transformation_GridToIndex(self, position):
        ind = position[1] * self.width + position[0]
        if (ind >= 0 and ind <= ((self.width * self.height) - 1)):
            self.var = int(ind)
            return True
        else:
            return False



    def transformation_IndexToGrid(self, index):
        y = index // self.width
        x = index % self.width

        if (x>=0 and x <= self.width - 1 and y >= 0 and y <= self.height - 1):
            self.var = np.array([int(x), int(y)]) #output is an pose 2D
            return True
        else: return False


    def getRandomFrontier(self, request):
        """ Return random frontier """
        print("You are in Random frontier")
        # TODO
        self.Updated = True
        try :
             frontiers = self.computeWFD()
             frontier = np.random.choice(frontiers)
        except :
             frontier = []     
        
        print("frontiers random choice , " ,frontiers)
        #frontier = np.random.choice(frontiers) #select random frontier
        print("frontiers random choice 2 , " ,frontier)
        centerOfFrontier = 0  # TODO: compute center of the randomly drawn frontier here
        sort, count = self.dismember_frontiers(frontiers)
      
        for z in range(len(sort)):
            if frontier in sort[z]:
                centerOfFrontier = count[z]
        #print(centerOfFrontier)
        position = self.transform_GridToCoordinates(np.array(centerOfFrontier))

        x, y = round(position[0],1), round(position[1],1) # TODO: transform the coordinates from grid to real-world coordinates (in meters)
        res = GenerateFrontierResponse(Pose2D(x, y, 0.0))

        return res

    def getClosestFrontier(self, request):
        """ Return frontier closest to the robot """
         # TODO
        print("You are closest Frontier")
        pose = self.getRobotCoordinates()
        self.definetly_Updated = False
        self.Updated = True
        while (not self.definetly_Updated): pass
        self.definetly_Updated = False

        if (len(self.frontiersList) == 0):
            try:
                frontiers = self.computeWFD()
                if (len(frontiers)>0):
                    sort, count = self.dismember_frontiers(frontiers)
                    for number in count:
                        print(self.transform_GridToCoordinates(np.array(number)))

                    self.frontiersList = count
                    #print("ESTIMATED FRONTS: ", len(self.frontiersList))
                else: self.frontiersList = []

            except:
                self.frontiersList = []    
        max = 999
        min = 0
        response = GenerateFrontierResponse(Pose2D(-999, -999, 0.0))
       
        if (len(self.frontiersList)):
            Remove = np.array(())
            #print("Currnently have:", len(self.frontiersList))
            var2 = []
            for p in self.frontiersList:
                if (not self.is_it_correct(p)): continue
                else: var2.append(p)
                tmpVar = self.heur(pose, p)
                
                if tmpVar <= max:
                    min = self.transform_GridToCoordinates(np.array(p))
                    max = tmpVar
                    Remove = p
            #print(max, len(self.frontiersList))
            var = []
            for p in var2:
                if p[0] != Remove[0] and p[1] != Remove[1]:
                    var.append(p)
            self.frontiersList = var

            try:
                x, y = round(min[0],5), round(min[1],5)  # TODO: compute the index of the best frontier
                print("Closest Frontier: ", x, y)
                response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
                return response
            except:
                response = self.getClosestFrontier(request)
                return response
        return response

    def is_it_correct(self, loc):
        # check if it is valid frontier
        close_dist = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]
        cnt = Counter()
        for k in range(0,int(5/(self.res) + 1)):
            for x in close_dist:
                try:
                    if self.grid2[loc[0]+x[0]][loc[1]+x[1]] == -1: cnt["unk"] += 1
                    elif self.grid2[loc[0]+x[0]][loc[1]+x[1]] == 0: cnt["empty"] += 1
                    else: cnt["obs"] += 1
                except:
                    pass
        print(loc, self.transform_GridToCoordinates(np.array(loc)), cnt["unk"], cnt["obs"], float(sum(cnt.values())))
        if (cnt["empty"]/float(sum(cnt.values())) > 0.8 or cnt["unk"]/float(sum(cnt.values())) <= 0.275 or (cnt["unk"] + cnt["obs"])/float(sum(cnt.values()))  >= 0.95): return False
        else: return True 
        

    def getRobotCoordinates(self):
        """ Get the current robot position in the grid """
        #print('Calculating robot coordinates')
        try:
            trans = self.tfBuffer.lookup_transform(self.mapFrame, self.robotFrame, rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Cannot get the robot position!")
            self.robotPosition = None
        else:
            position = np.array((trans.transform.translation.x, trans.transform.translation.y)).reshape(2,)
            print("pose : ",position)
            tfGrid = self.transform_robotPositionToGrid(position)
            return tfGrid  # TODO: transform the robot coordinates from real-world (in meters) into grid
            
            
    def transform_robotPositionToGrid(self, pose):
        robotPose = ((self.matrix_R.T).dot(pose - self.matrix_t)) / self.res  # TODO: transform the robot coordinates from real-world (in meters) into grid
        robotPose[0] = round(robotPose[0])
        robotPose[1] = round(robotPose[1])
        robotPose = np.array((robotPose), dtype=int)
        return robotPose

    def transform_GridToCoordinates(self, pose):
        final = (self.matrix_R.dot(pose * self.res) + self.matrix_t)
        return final
            
    def check_if_nbrs(self, list, pnt):

        close_dist = [1, self.width, self.width-1, self.width+1, -1, -self.width, -self.width-1, -self.width+1]
        for l in list:
            diff = abs(l-pnt)
            for d in close_dist:
                if d == diff: return True
        return False

    def dismember_frontiers(self,frontiers):
        var = []
        fnl = []
        var.append(frontiers[0])
        for n in range(1, len(frontiers)):
            if self.check_if_nbrs(var, frontiers[n]):
                var.append(frontiers[n])
            else:
                fnl.append(var)
                var = []
                var.append(frontiers[n])
        if (len(var)): fnl.append(var)
        print("Frontier groups:")
        print(len(fnl))
        c = []
        for m in fnl:
            x_max = 0
            y_max = 0
            x_min = 99999
            y_min = 99999
            for q in m:
                self.transformation_IndexToGrid(q)
                if x_max <= self.var[0]: x_max = self.var[0]
                if x_min >= self.var[0]: x_min = self.var[0]
                if y_max <= self.var[1]: y_max = self.var[1]
                if y_min >= self.var[1]: y_min = self.var[1]
            c.append(np.array(([int(round((x_max+x_min)/2)), int(round((y_max+y_min)/2))])))
        #print("count:",c)
        pnts = []
        for f in fnl:
            for w in f:
                pnts.append(w)
        return fnl, c

    def heur(self, start, target):
        #euclidian distance
        return (math.sqrt((start[0] - target[0])**2 + (start[1] - target[1])**2))
    
   
    def extractGrid(self, msg):
            # TODO: extract grid from msg.data and other usefull information
        self.grid = msg.data
        self.gridInfo = msg.info
        self.res = msg.info.resolution
        self.width = msg.info.width
        self.height = msg.info.height
        self.origin = msg.info.origin
        self.position = msg.info.origin.position
        self.orientation = msg.info.origin.orientation
        self.grid2 = np.reshape(msg.data, (self.height, self.width)).T
        self.Updated = False
        self.definetly_Updated = True


    def grid_cb(self, msg):
       
        if (self.Updated) : self.extractGrid(msg)
        if not self.gridReady:
            
            # TODO: Do some initialization of necessary variables
            theta = round(round(self.orientation.z, 3) * 2, 2)
            cs, sn = np.cos(theta), np.sin(theta)
            self.matrix_R = np.array(((cs, -sn), (sn, cs)))
            self.matrix_t = (np.array((self.position.x, self.position.y))).reshape(2, )

            
            self.grf_service = rospy.Service('get_random_frontier', GenerateFrontier, self.getRandomFrontier)
            self.gcf_service = rospy.Service('get_closest_frontier', GenerateFrontier, self.getClosestFrontier)
            self.gridReady = True
            


if __name__ == "__main__":
    fe = FrontierExplorer()
    rospy.loginfo('Lowest point.')
    rospy.spin()
