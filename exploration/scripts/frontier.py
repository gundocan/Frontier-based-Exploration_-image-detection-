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
from queue import *
import tf2_ros
from collections import Counter
import math

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
        self.tmp = 0
        self.grid = []
        self.grid2 = []
        self.definetly_Updated= False
        self.Updated = True
        self.gridInfo = []
        self.frontiersList = []
        # Get some useful parameters
        self.lookAroundSteps = 0
        self.mapFrame = rospy.get_param("~map_frame", "map")
        self.robotFrame = rospy.get_param("~robot_frame", "robot")
        self.robotDiameter = float(rospy.get_param("~robot_diameter", 0.2))
        self.occupancyThreshold = int(rospy.get_param("~occupancy_threshold", 10))
        #self.odo=rospy.Subscriber('/odom',Odometry,self.getRobotCoordinates)
        # Helper variable to determine if grid was received at least once
        self.gridReady = False
        # You may wish to listen to the transformations of the robot
        self.tfBuffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.R = np.empty((2, 2))
        self.t = np.empty((2,))
        # Subscribe to grid
        self.gridSubscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)
        rospy.loginfo('Frontier initialized.')
        # TODO: you may wish to do initialization of other variables
    
    def WFD(self,robotInd):
       # print("wfd algortyhm")
        qM = Queue()
        mol = []
        mcl = []
        fol = []
        fcl = []
        status = {}
        for  i in range (0, self.width * self.height):
             status[i] = "none"
        frontiers = []
        qM.put(robotInd)
        status[robotInd] = "mol"
        dist = [1, self.width, -1, -self.width, self.width - 1, self.width + 1, -self.width + 1, -self.width - 1]
        while not qM.empty() :
            p=qM.get()

            if (status[p] == "mcl"): continue

            if self.indToGrid(p) and self.isFrontier(p) :
                qF = Queue()
                nf = []
                qF.put(p)
                status[p] = "fol"

                while not qF.empty() :
                    q =qF.get()
                    if(status[q] == "mol" or status[q] == "fcl"): continue
                    if (self.indToGrid(q) and self.isFrontier(q)):
                        nf.append(q)
                        for k in dist :
                            if (self.indToGrid(q + k) and status[q + k] != "fol" and status[q + k] != "fcl" and status[q + k] != "mcl"):
                                qF.put(q+k)
                                status[q + k] = "fol"

                    status[q] = "fcl"

                if len(nf) >= 1:
                    for i in nf:
                       # print(len(nf), "frontier points added")
                        frontiers.append(i)
                        status[i] = "mcl"

            for v in dist :
                if(self.indToGrid(p+v)) :
                    if (status[v+p]!= "mol" and status[v+p] != "mcl") :
                        for x in dist:
                            if(self.indToGrid(v+p+x) and (self.grid[p+v+x] == 0)) :
                                qM.put((p+v))
                                status[p+v]= "mol"
                                break
            status[p]= "mcl"

        return frontiers







    def computeWFD(self):
        """ Run the Wavefront detector """
        print("computing WFD ")
        frontiers = []
        # TODO: First, you should try to obtain the robots coordinates
          # TODO: Then, copy the occupancy grid into some temporary variable and inflate the obstacles
        pos = self.getRobotCoordinates()
        robotInd = 0
        if (self.gridToInd(pos)): robotInd = self.tmp

        frontiers = self.WFD(robotInd)
        print("Frontiers : ",frontiers) # TODO: Run the WFD algorithm - see the presentation slides for details on how to implement it
        
        return frontiers
        
    

    def isFrontier(self, ind):
        if (self.grid[ind] != -1): return False
        if (ind%self.width == 0): dist = [1, self.width, self.width + 1, -self.width, -self.width + 1]
        elif (ind%self.width == self.width-1): dist = [-1, self.width, self.width - 1, -self.width, -self.width - 1]
        else: dist = [1, self.width, -1, -self.width, self.width - 1, self.width + 1, -self.width + 1, -self.width - 1]
        #toReturn = True
        for k in dist:
            if self.indToGrid(ind+k):
                if self.grid[ind+k] == 0: return True
        return False


    def gridToInd(self, pos):
        result = pos[1] * self.width + pos[0]
        if (result >= 0 and result <= ((self.width * self.height) - 1)):
            self.tmp = int(result)
            return True
        else:
            return False



    def indToGrid(self, ind):
        x = ind%self.width
        y = ind//self.width
        if (x>=0 and x <= self.width - 1 and y >= 0 and y <= self.height - 1):
            self.tmp = np.array([int(x), int(y)])
            return True
        else: return False


    def getRandomFrontier(self, request):
        """ Return random frontier """
        print("random frontierdasin")
        # TODO
        self.Updated = True
        frontiers = self.computeWFD()
        frontier = np.random.choice(frontiers)

        frontierCenter = 0  # TODO: compute center of the randomly drawn frontier here
        grp, cnt = self.degroupF(frontiers)
      
        for k in range(len(grp)):
            if frontier in grp[k]:
                frontierCenter = cnt[k]
        #print(frontierCenter)
        pos = self.gridtoCoord(np.array(frontierCenter))

        x, y = round(pos[0],1), round(pos[1],1) # TODO: transform the coordinates from grid to real-world coordinates (in meters)
        response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
       # print(response)
        return response

    def getClosestFrontier(self, request):
        """ Return frontier closest to the robot """
         # TODO
        
        pos = self.getRobotCoordinates()
        self.definetly_Updated = False
        self.Updated = True
        while (not self.definetly_Updated): pass
        self.definetly_Updated = False
        #print("HERE")
        if (len(self.frontiersList) == 0):
            try:
                print("FINDING NEW FRONTS")
                #self.Update = True
                """ Return frontier closest to the robot """
                # TODO
                frontiers = self.computeWFD()
                #print("here2")
                if (len(frontiers)>0):
                    grp, cnt = self.degroupF(frontiers)
                    for c in cnt:
                        print(self.gridtoCoord(np.array(c)))

                    self.frontiersList = cnt
                    print("FRONTS FOUND: ", len(self.frontiersList))
                else: self.frontiersList = []
                #print("here4")
            except:
                self.frontiersList = []    
        min = 999
        minCnt = 0
        #print(cnt)
        response = GenerateFrontierResponse(Pose2D(-999, -999, 0.0))
       
        if (len(self.frontiersList)):
            toRemove = np.array(())
            print("Currnently have:", len(self.frontiersList))
            tmp2 = []
            for n in self.frontiersList:
                if (not self.isValid(n)): continue
                else: tmp2.append(n)
                tmpVal = self.heuristic(pos, n)
                
                if tmpVal <= min:
                    minCnt = self.gridtoCoord(np.array(n))
                    min = tmpVal
                    toRemove = n
            print(min, len(self.frontiersList))
            tmp = []
            for n in tmp2:
                if n[0] != toRemove[0] and n[1] != toRemove[1]:
                    tmp.append(n)
            self.frontiersList = tmp
            print("And now... :", len(self.frontiersList))
            print(self.frontiersList)
            print("minCnt", minCnt)
            try:
                x, y = round(minCnt[0],5), round(minCnt[1],5)  # TODO: compute the index of the best frontier
                print("CF: ", x, y)
                response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
                return response
            except:
                response = self.getClosestFrontier(request)
                return response
        return response

    def isValid(self, p):

        dist = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]
        c = Counter()
        for k in range(0,int(5/(self.res) + 1)): #1
            for d in dist:
                try:
                    if self.grid2[p[0]+d[0]][p[1]+d[1]] == -1: c["unknown"] += 1
                    elif self.grid2[p[0]+d[0]][p[1]+d[1]] == 0: c["free"] += 1
                    else: c["obstacle"] += 1
                except:
                    pass
        print(p, self.gridtoCoord(np.array(p)), c["unknown"], c["obstacle"], float(sum(c.values())))
        if (c["free"]/float(sum(c.values())) > 0.8 or c["unknown"]/float(sum(c.values())) <= 0.275 or (c["unknown"] + c["obstacle"])/float(sum(c.values()))  >= 0.95): return False 
        else: return True 
        

    def getRobotCoordinates(self):
        """ Get the current robot position in the grid """
        print('Calculating robot coordinates')
        try:

          
            trans = self.tfBuffer.lookup_transform(self.mapFrame, self.robotFrame, rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Cannot get the robot position!")
            self.robotPosition = None
        else:
            pos = np.array((trans.transform.translation.x, trans.transform.translation.y)).reshape(2,)
            print("pose : ",pos)
            tfGrid = self.RobotPosToGrid(pos)
            #print("Gride girdi: ", tfGrid)
            return tfGrid  # TODO: transform the robot coordinates from real-world (in meters) into grid
            
            
    def RobotPosToGrid(self, pos):
        robotPos = ((self.R.T).dot(pos-self.t))/ self.res  # TODO: transform the robot coordinates from real-world (in meters) into grid
        robotPos[0] = round(robotPos[0])
        robotPos[1] = round(robotPos[1])
        robotPos = np.array((robotPos), dtype=int)
        return robotPos

    def gridtoCoord(self, pos):
        ret = (self.R.dot(pos*self.res) + self.t)
       
        return ret
            
    def isNeighbours(self, arr, p):
        #dist = [1, 50, 49, 51, 50]
        dist = [1, self.width, self.width-1, self.width+1, -1, -self.width, -self.width-1, -self.width+1]
        for a in arr:
            dif = abs(a-p)
            for x in dist:
                if x == dif: return True
        return False

    def degroupF(self,frontiers):
        tmp = []
        final = []
        tmp.append(frontiers[0])
        for x in range(1, len(frontiers)):
            if self.isNeighbours(tmp, frontiers[x]):
                tmp.append(frontiers[x])
            else:
                final.append(tmp)
                tmp = []
                tmp.append(frontiers[x])
        if (len(tmp)): final.append(tmp)  
        print("Frontier groups:")
        print(len(final))    
        cnt = []
        for p in final:
            xmax = 0
            ymax = 0
            xmin = 99999
            ymin = 99999
            for k in p:
                self.indToGrid(k)
                if xmax <= self.tmp[0]: xmax = self.tmp[0]
                if xmin >= self.tmp[0]: xmin = self.tmp[0]
                if ymax <= self.tmp[1]: ymax = self.tmp[1]
                if ymin >= self.tmp[1]: ymin = self.tmp[1]
            cnt.append(np.array(([int(round((xmax+xmin)/2)), int(round((ymax+ymin)/2))]))) 
        print("count:",cnt)       
        points = []    
        for z in final:
            for g in z:
                points.append(g)
        return final, cnt        

    def heuristic(self, start, target):
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
            c, s = np.cos(theta), np.sin(theta)
            self.R = np.array(((c, -s), (s, c)))
            self.t = (np.array((self.position.x, self.position.y))).reshape(2, )
           # print("R: ",self.R)
           # print("t : ",self.t)
            # Create services
            
            self.grf_service = rospy.Service('get_random_frontier', GenerateFrontier, self.getRandomFrontier)
            self.gcf_service = rospy.Service('get_closest_frontier', GenerateFrontier, self.getClosestFrontier)
            self.gridReady = True
            


if __name__ == "__main__":
    fe = FrontierExplorer()
    rospy.loginfo('Lowest point.')
    rospy.spin()

