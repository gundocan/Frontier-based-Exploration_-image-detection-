#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PointStamped, Pose
import lxml.etree as ET
import numpy as np
import rospkg
import os
from uuid import uuid4
import roslaunch
import cv2
from aro_evaluathor.utils import convertMap, MAP_SIZE
import tf
import csv
from datetime import datetime
import re
from collections import deque


class Evaluathor():
    EMPTY_THRESHOD = 25
    D_PLUS = (0, 255, 0)  # GREEN: (true empty identified as empty)
    D_MINUS = (0, 0, 255)  # RED: -1 point (true occupied or unobservable identified as empty)
    D_PAD = (0, 0, 0)  # BLACK: padded cells
    D_GOOD = (255, 0, 0)  # BLUE: correctly identified occupied or unobservable cells
    D_MISSED = (0, 170, 242)  # ORANGE: true occupied unobserved
    D_OTHER = (148, 255, 255)  # YELLOW: other mismatches

    BARBIE_SCORE_ERROR = -5000
    BARBIE_SCORE_GOOD = 5000
    BARBIE_SCORE_EXCELLENT = 10000
    BARBIE_TH_ERROR = 1.0
    BARBIE_TH_EXCELLENT = 0.5

    RUN_SINGLE = "single"
    RUN_MANUAL = "manual"
    RUN_AUTO = "auto"

    MAP_TOPIC = "/occupancy"
    BARBIE_TOPIC = "/barbie_point"

    BARBIE_BUFFER_SIZE = 3

    def __init__(self):
        rospack = rospkg.RosPack()
        self.aro_sim_pkg = rospack.get_path("aro_sim")  # aro_sim package path
        self.aro_eval_pkg = rospack.get_path("aro_evaluathor")  # aro_eval package path
        self.outFolder = os.path.expanduser("~/aro_evaluation")
        if not os.path.isdir(self.outFolder):
            os.mkdir(self.outFolder)

        self.mapImage = None  # current map image, i.e. the received occupancy grid
        self.requestedMap = rospy.get_param("~map_name", "unknown" + uuid4().hex)  # name of the requested world
        self.multiEval = type(self.requestedMap) is list  # whether multiple maps are evaluated
        self.spawnMode = rospy.get_param("~spawn_mode", "fixed")  # random starting position
        self.runMode = rospy.get_param("~run_mode", self.RUN_MANUAL)  # if run o multiple maps, how are the maps being switched
        self.timeLimit = rospy.get_param("~time_limit", 180)  # go to next map after X seconds if run mode is auto
        self.timeInterval = rospy.get_param("~time_interval", 30)  # evaluation interval

        self.sim_launch = None  # variable to hold the simulator launcher
        self.mapIndex = -1  # for running on multiple maps from a list (i.e. requestedMap is a list)
        self.poseIndex = -1  # for init pose from a list
        self.mapFrame, self.odomFrame = "map", "odom"
        self.initPoses = None
        self.startTime = None

        self.stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # compute data fields
        self.intervalFields = ["score_{}".format(t) for t in np.r_[self.timeInterval:self.timeLimit+1:self.timeInterval]]
        self.dataFields = ["map", "run", "score"] + self.intervalFields + ["per run", "final score"]
        self.data = []

        self.mapListener = rospy.Subscriber(self.MAP_TOPIC, OccupancyGrid, self.mapUpdate_cb, queue_size=1)
        self.barbieListener = rospy.Subscriber(self.BARBIE_TOPIC, PointStamped, self.barbie_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration.from_sec(0.05), self.timer_cb)

        rospy.loginfo("Setting up evaluation:\n\t{} map mode\n\tmap(s): {}\n\t{} run mode\n\t{} spawn mode".format(
            "multi" if self.multiEval else "single",
            self.requestedMap,
            self.runMode,
            self.spawnMode
        ))

    def __padImage2Shape(self, image, shape, value=-2):
        """Pads the input image to the specified shape. Make sure
        that the shape of the input image is at most the size of desired shape.

        Arguments:
            image {ndarray} -- image to be padded
            shape {tuple} -- desired shape of the image

        Keyword Arguments:
            value {int} -- value to pad the extra shape with (default: {-1})

        Returns:
            ndarray
        """
        extra = np.r_[shape] - image.shape
        borders = np.round(extra / 2)
        borders = np.r_["0,2,1", np.ceil(extra - borders), borders].astype(int).T.ravel()
        return cv2.copyMakeBorder(image, *borders, borderType=cv2.BORDER_CONSTANT, value=value)

    def __putText(self, image, text, origin, size=1, color=(255, 255, 255), thickness=2):
        """ Prints text into an image. Uses the original OpenCV functions
        but simplifies some things. The main difference betwenn OpenCV and this function
        is that in this function, the origin is the center of the text.

        Arguments:
            image {ndarray} -- the image where the text is to be printed into
            text {str} -- the text itself
            origin {tuple} -- position in the image where the text should be centered around

        Keyword Arguments:
            size {int} -- size/scale of the font (default: {1})
            color {tuple} -- color of the text (default: {(255, 255, 255)})
            thickness {int} -- line thickness of the text (default: {2})

        Returns:
            ndarray -- the original image with the text printed into it
        """
        offset = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0] * np.r_[-1, 1] / 2
        return cv2.putText(image, text, tuple(np.int32(origin + offset).tolist()), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    def __formatTime(self, secs):
        """Splits the time in seconds into hours, minutes, and seconds

        Arguments:
            secs {int} -- time in seconds

        Returns:
            tuple
        """
        h = int(secs / 3600)
        r = secs - h * 3600
        m = int(r / 60)
        r -= m * 60
        return h, m, int(r)

    def __generateRandomSpawn(self):
        """ Generates random x & y position for the robot.
        """
        return np.random.randn(2) / 2

    def mapUpdate_cb(self, msg):
        """ Callback for the occupancy grid messages.
        """
        self.mapInfo = msg.info
        # startTime=None -> simulation restart, time counter starts at the first occupancy grid msg
        if self.startTime is None:
            self.startTime = rospy.Time.now()
            rospy.loginfo(self.startTime)

        try:
            # reshape the map into the correct size
            self.mapImage = np.reshape(msg.data, (self.mapInfo.height, self.mapInfo.width))
        except Exception as e:
            rospy.logerr(e)

    def timer_cb(self, tim):
        try:
            # compare the received map with the GT
            self.compareMaps()
        except Exception as e:
            rospy.logerr(e)

    def evalBarbie(self, log=True):
        if len(self.barbieBuffer) == 0:
            self.barbieScore = 0
            return

        diff = min(self.barbieBuffer)
        if diff > self.BARBIE_TH_ERROR:
            message = "Oh, applesauce! Your best barbie position estimate is way off, you currently have {} points :'(".format(self.BARBIE_SCORE_ERROR)
            self.barbieScore = self.BARBIE_SCORE_ERROR
        elif diff < self.BARBIE_TH_EXCELLENT:
            message = "An abso-freaking-lutely excellent barbie detection, you have {} points! Bravo!".format(self.BARBIE_SCORE_EXCELLENT)
            self.barbieScore = self.BARBIE_SCORE_EXCELLENT
        else:
            message = "Your estimation of barbie position is good, which gets you {} points. But you could do better, I believe in you!".format(self.BARBIE_SCORE_GOOD)
            self.barbieScore = self.BARBIE_SCORE_GOOD
        if log:
            rospy.loginfo(message)

    def barbie_cb(self, msg):
        try:
            p = msg.point
            position = np.r_[[getattr(p, f) for f in "xyz"]]
            diff = np.linalg.norm(position - self.barbieGT)
        except Exception as e:
            rospy.logerr("Received a barbie position but there was an error during processing:\n {}\nIgnoring barbie position.".format(e))
            return

        rospy.loginfo("Received a barbie position @ {}. The distance to GT is {:4f} m.".format(position, diff))
        self.barbieBuffer.append(diff)
        self.evalBarbie()

    def compareMaps(self):
        """Compares the self.mapImage and self.gtImage variables.
        """
        if self.mapImage is None:
            rospy.logwarn("Map Image not received yet!")
            return False

        angle = tf.transformations.euler_from_quaternion([self.mapInfo.origin.orientation.x, self.mapInfo.origin.orientation.y, self.mapInfo.origin.orientation.z, self.mapInfo.origin.orientation.w])[2] - self.gtAngle
        shift = (np.r_[self.mapInfo.origin.position.x, self.mapInfo.origin.position.y] + self.startingPose - self.gtShift) / self.mapInfo.resolution + np.r_[1, 1]
        center = tuple(np.r_[self.mapImage.shape] / 2)
        rMat = cv2.getRotationMatrix2D(center, np.rad2deg(angle), 1)
        tMat = np.diag([1, 1, 1]).astype(np.float)
        tMat[:2, 2] = shift
        transMatrix2D = np.dot(tMat, np.vstack((rMat, [0, 0, 1])))
        transformedMapImage = cv2.warpAffine(self.mapImage.astype(float).T, transMatrix2D[:2, :], self.mapImage.shape, flags=cv2.INTER_NEAREST, borderValue=-2).T

        # reshape the grids to the same maximal size
        finalShape = np.max((transformedMapImage.shape, self.gtImage.shape), axis=0)
        mapPad = self.__padImage2Shape(transformedMapImage, finalShape)
        gtPad = self.__padImage2Shape(self.gtImage, finalShape)

        # create clones and set them to values:
        # 0 = empty
        # 1 = occupied
        # -1 = unknown
        # -2 = padded
        mapConv = mapPad.copy()
        gtConv = gtPad.copy()

        # compute empty vs occupied cells
        mapConv[np.where(np.logical_and(mapPad < self.EMPTY_THRESHOD, mapPad >= 0))] = 0
        gtConv[np.where(np.logical_and(gtPad < self.EMPTY_THRESHOD, gtPad >= 0))] = 0
        mapConv[np.where(mapPad >= self.EMPTY_THRESHOD)] = 1
        gtConv[np.where(gtPad >= self.EMPTY_THRESHOD)] = 1

        # compute difference grid
        diffGrid = np.zeros(np.r_[finalShape, 3], dtype=np.uint8)

        # plus -> map == empty & gt == empty
        plus = np.where(np.logical_and(mapConv == 0, gtConv == 0))
        # minus -> map == empty & (gt == occup | unknown)
        minus = np.where(np.logical_and(mapConv == 0, np.logical_or(gtConv == 1, gtConv == -1)))
        # good -> (map == occup & gt == occup) | (map == unknown & gt == unknown)
        good = np.where(np.logical_or(
            np.logical_and(mapConv == 1, gtConv == 1),
            np.logical_and(mapConv == -1, gtConv == -1)))

        diffGrid[plus] = self.D_PLUS  # primary "plus one point" color
        diffGrid[minus] = self.D_MINUS  # primary "minus one point" color
        diffGrid[good] = self.D_GOOD
        # map == unknown & gt == occup
        diffGrid[np.where(np.logical_and(mapConv == -1, gtConv == 1))] = self.D_MISSED
        diffGrid[np.where(np.logical_or(
            np.logical_and(mapConv == 1, np.logical_or(
                gtConv == 0,
                gtConv == -1)),
            np.logical_and(mapConv == -1, gtConv == 0)))] = self.D_OTHER
        # either map or gt == padded (i.e. -> ignore)
        diffGrid[np.where(np.logical_or(mapConv == -2, gtConv == -2))] = self.D_PAD

        # combine all maps into one image
        # diffShape = np.r_[diffGrid.shape[:2]]
        output = np.hstack((
            convertMap(mapPad) * 255,
            convertMap(gtPad) * 255,
            cv2.resize(np.flipud(diffGrid), (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_NEAREST)
            )).astype(np.uint8)

        # compute score
        mappingScore = -np.count_nonzero(np.logical_not(np.logical_or(mapConv == -2, mapConv == gtConv)))
        actualScore = mappingScore + self.barbieScore

        # compute time
        if self.startTime is None:
            currentTime = rospy.Time.from_sec(0).secs
        else:
            currentTime = (rospy.Time.now() - self.startTime).secs

        # make text
        topScreen = np.zeros((32, output.shape[1], 3), dtype=np.uint8)
        xp = MAP_SIZE / 2
        yp = 12
        # add image captions
        topScreen = self.__putText(topScreen, "computed occupancy grid", (xp, yp))
        topScreen = self.__putText(topScreen, "GT ({})".format(self.mapName), (xp * 3, yp))
        topScreen = self.__putText(topScreen, "difference", (xp * 5, yp))

        # add scores
        scoreScreen = np.zeros((128, output.shape[1], 3), dtype=np.uint8)
        scoreScreen = self.__putText(scoreScreen, "score = {:5d}".format(mappingScore), (xp, 14), color=(0, 255, 0), size=0.8)
        scoreScreen = self.__putText(scoreScreen, "score (+barbie) = {:5d}".format(actualScore), (xp, 42), color=(0, 128, 0), size=0.6, thickness=1)

        gray = 196
        scoreScreen = self.__putText(scoreScreen, "mapping score= {:5d}".format(mappingScore), (xp * 3, 10), color=(gray, gray, gray), size=0.6, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "barbie score = {:5d}".format(self.barbieScore), (xp * 3, 32), color=(gray, gray, gray), size=0.6, thickness=1)

        # add time
        scoreScreen = self.__putText(scoreScreen, "time: {:2d}:{:02d}:{:02d}".format(*self.__formatTime(currentTime)), (xp * 5, 16))

        # add legend
        scoreScreen = self.__putText(scoreScreen, "Legend: ", (xp, 85), size=0.5, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "(press 'q' to exit)", (xp, 110), color=(gray, gray, gray), size=0.4, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "true empty & marked empty", (xp * 3, 85), size=0.5, color=self.D_PLUS, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "(true occup | true unobs) & marked empty", (xp * 3, 102), size=0.5, color=self.D_MINUS, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "other correctly marked", (xp * 3, 118), size=0.5, color=self.D_GOOD, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "true occup & marked unobs", (xp * 5, 85), size=0.5, color=self.D_MISSED, thickness=1)
        scoreScreen = self.__putText(scoreScreen, "other incorrectly marked", (xp * 5, 104), size=0.5, color=self.D_OTHER, thickness=1)

        # stack top and score screens to the map images
        output = np.vstack((topScreen, output, scoreScreen))
        self.videoWriter.write(output)

        # show output
        cv2.imshow("Output", output)
        key = cv2.waitKey(10) & 0xFF

        if (currentTime > 0 and ((currentTime % self.nextEvalTime) == 0)) or currentTime >= self.timeLimit:
            # store current score
            self.data[-2][self.intervalFields[np.round(currentTime / self.timeInterval).astype(np.int32) - 1]] = mappingScore
            rospy.loginfo("Scoring interval <{}..{}>s had passed, writing current score:\n"
                          "\ttotal = {}\n\t\tmapping = {}\n\t\tbarbie = {}".format(self.nextEvalTime - self.timeInterval, self.nextEvalTime,
                                                                                   actualScore, mappingScore, self.barbieScore))
            self.nextEvalTime += self.timeInterval

        if currentTime >= self.timeLimit:
            pointsMapping = [self.data[-2][ifield] for ifield in self.intervalFields]
            averageMappingPoints = np.round(np.average(pointsMapping))
            self.data[-2]["per run"] = averageMappingPoints
            self.evalBarbie(log=False)
            self.data[-1]["per run"] = self.barbieScore
            pointsSum = averageMappingPoints + self.barbieScore
            self.data.append({"score": "sum", "per run": pointsSum})
            if self.runMode == self.RUN_SINGLE:
                self.stopSim()
                rospy.loginfo("Time limit reached, stopping.")
                rospy.signal_shutdown("End of evaluation.")
                return
            else:
                rospy.loginfo("Time limit reached, restarting.")
                self.restart()

        elif key == ord("q"):  # quit simulation
            self.stopSim()

    def __loadMap(self, mapName):
        self.mapName = mapName
        # load map GT
        mapFile = os.path.join(self.aro_eval_pkg, "maps", "{}.txt".format(self.mapName))
        poseFile = os.path.join(self.aro_eval_pkg, "init_poses", "{}.txt".format(self.mapName))

        if not os.path.exists(mapFile):
            e_msg = "Ground truth for the world {} was not found!".format(self.mapName)
            rospy.logfatal(e_msg)
            raise IOError(e_msg)
        if not os.path.exists(poseFile):
            e_msg = "Initial poses for the world {} was not found!".format(self.mapName)
            rospy.logfatal(e_msg)
            raise IOError(e_msg)

        launchFile = os.path.join(self.aro_sim_pkg, "launch", "turtlebot3.launch")
        tree = ET.parse(launchFile)
        root = tree.getroot()
        found = False
        for elm in root.iter("arg"):
            attributes = elm.attrib
            if "barbie_args" in attributes.itervalues():
                if "if" in attributes and self.mapName in attributes["if"]:
                    self.barbieGT = np.r_[re.findall(r"(?<=[xyz])(\s*\-?[\d.]+)", attributes["default"])].astype(np.float)
                    found = True
                    break
        if not found:
            self.barbieGT = np.random.randn(3) * 1000

        self.gtImage = np.loadtxt(mapFile)
        meta = self.gtImage[-1, :]
        self.gtShift = meta[:2]
        self.gtAngle = tf.transformations.euler_from_quaternion(meta[3:7])[2]
        self.gtImage = self.gtImage[:-1, :]

        # create empty map
        self.mapImage = np.ones_like(self.gtImage) * -1
        emptyOrigin = Pose()
        emptyOrigin.position.x, emptyOrigin.position.y = meta[:2]
        self.mapInfo = MapMetaData(resolution=0.05, origin=emptyOrigin)

        self.initPoses = np.loadtxt(poseFile)
        if self.initPoses.ndim == 1:
            self.initPoses = self.initPoses[np.newaxis]

        # compute GT empty and impassable counts
        emptyGT = np.where(np.logical_and(self.gtImage >= 0, self.gtImage < self.EMPTY_THRESHOD))
        self.gtEmptyCount = float(np.shape(emptyGT)[1])
        impassableGT = np.where(np.logical_or(self.gtImage >= self.EMPTY_THRESHOD, self.gtImage == -1))
        self.gtImpassableCount = float(np.shape(impassableGT)[1])

    def stopSim(self):
        self.sim_launch.shutdown()
        self.videoWriter.release()

    def restart(self):
        if self.sim_launch is not None:
            self.stopSim()

        self.startTime = None
        self.nextEvalTime = self.timeInterval
        self.barbieScore = 0
        self.barbieBuffer = deque(maxlen=self.BARBIE_BUFFER_SIZE)

        if "random" in self.spawnMode:
            # random spawn location
            self.startingPose = self.__generateRandomSpawn()
            spawn_command = ["spawn_args:=-x {:.4f} -y {:.4f} -z 0.0".format(*self.startingPose)]
            if self.spawnMode == "random_once":  # spawn random once and then set back to fixed
                self.spawnMode = "fixed"
        elif "list" in self.spawnMode:
            self.poseIndex += 1
            if self.initPoses is None or self.poseIndex == self.initPoses.shape[0]:  # end of init poses list
                # goto next map
                if self.multiEval:
                    # sequential map list evaluation mode
                    # automatically go through all maps until the last one
                    self.mapIndex += 1
                    if self.mapIndex == len(self.requestedMap):
                        # should be the end of evaluation
                        rospy.signal_shutdown("End of evaluation.")
                        return
                    self.__loadMap(self.requestedMap[self.mapIndex])
                    self.poseIndex = 0
                else:  # single map mode
                    self.poseIndex = 0
                    self.__loadMap(self.requestedMap)
                self.data.append({"map": self.mapName})

            self.startingPose = self.initPoses[self.poseIndex, :]
            spawn_command = ["spawn_args:=-x {:.4f} -y {:.4f} -z 0.0".format(*self.startingPose)]
        else:
            spawn_command = []

        self.data.append({"run": self.poseIndex, "score": "mapping"})
        self.data.append({"score": "detection"})

        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        vidFName = "vid_{}_{}.mp4".format(self.mapName, self.stamp) if self.runMode == self.RUN_SINGLE else "vid_{}_{}_{}.mp4".format(self.mapName, self.poseIndex, self.stamp)
        self.videoWriter = cv2.VideoWriter(os.path.join(self.outFolder, vidFName), fourcc, 10, (1200, 560))

        # Launch the simulator
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch_command = ["aro_sim",
                          "turtlebot3.launch",
                          "world:={}".format(self.mapName)
                          ]
        launch_command += spawn_command

        print("Starting at pose: x={:.4f} y={:.4f}".format(*self.startingPose))

        sim_launch_file = roslaunch.rlutil.resolve_launch_arguments(launch_command)[0]
        sim_launch_args = launch_command[2:]
        launch_files = [(sim_launch_file, sim_launch_args)]
        self.sim_launch = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
        rospy.loginfo(self.sim_launch.roslaunch_files)
        self.sim_launch.force_log = True
        self.sim_launch.start()
        rospy.loginfo("ARO SIM launched.")

        self.startTime = None

    def showStatistics(self):
        print(self.data)
        # save results
        resultFilePath = os.path.join(self.outFolder, "results_{}.csv".format(self.stamp))
        print("Saving results to : {}".format(resultFilePath))
        total_sum = 0
        with open(resultFilePath, "w") as f:
            writer = csv.DictWriter(f, self.dataFields)
            writer.writeheader()
            n = 0
            for row in self.data:
                writer.writerow(row)
                if "score" in row and row["score"] == "sum":
                    n += 1
                    total_sum += row["per run"]
            # write out sum
            if n > 0:
                final_score = int(np.round(total_sum))
                writer.writerow({"final score": final_score})
                print("\n>>> FINAL SCORE = {} <<<\n".format(final_score))

    def run(self):
        self.restart()
        try:
            rospy.spin()  # spin
        finally:
            self.showStatistics()
            cv2.destroyAllWindows()
            self.sim_launch.shutdown()  # stop the simulator


if __name__ == "__main__":
    rospy.init_node("evaluathor")

    evaluathor = Evaluathor()
    evaluathor.run()
