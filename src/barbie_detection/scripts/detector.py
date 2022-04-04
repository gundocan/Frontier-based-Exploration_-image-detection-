#!/usr/bin/env python
import sys, time
import numpy as np
# Ros
import math

import rospy
import ros_numpy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from PIL import Image as PILimage
import message_filters
# Pytorch
import torch
# Our library
#from exploration.msg import barbie
import network
import utils

publish_image = True

def to_tensor(image):
    # normalize
    mean = [118, 117, 117]
    std = [57, 58, 60]
    image = image.astype(np.float32)
    image = (image - mean) / std
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    return image

class barbie_detector:
    def __init__(self):
        rospy.init_node('barbie_detector')
        # topics where we publish
        self.cloud_in_laser_camera =np.array([])
        self.image_pub = rospy.Publisher("/barbie_detections", Image, queue_size=1)
        self.point_pub = rospy.Publisher("/barbie_point", PointStamped, queue_size=1)
        # subscribed topics
        self.image_subscriber = message_filters.Subscriber(rospy.get_param("~rgb_topic","/camera/rgb/image_raw"), Image)
        self.cam_info_subscriber = message_filters.Subscriber(rospy.get_param("~rgb_info_topic","/camera/rgb/camera_info"), CameraInfo)
        self.laser_subscriber = message_filters.Subscriber(rospy.get_param("~laser_topic", "/cloud"), PointCloud2)
        # time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber,  self.laser_subscriber, self.cam_info_subscriber], 1, 0.5)
        # tf buffer
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # network
        self.model = network.Net()
        #self.barbie_status ="Not Found"
        #self.barbie_status_pub = rospy.Publisher("barbie_status", String , queue_size=1 )
        weights_path = rospy.get_param('~weights_path', 'trained_weights')
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        rospy.loginfo("Weights loaded from path: %s", weights_path)
        self.det_treshhold = rospy.get_param('~detection_threshold', 7)
        # callback registration
        self.ts.registerCallback(self.callback)
        print("Detector initialized")


    def callback(self, image_data, laser_data, cam_info):

        det = False
        # direct conversion to numpy - image
        np_data = np.fromstring(image_data.data, np.uint8)
        in_image = np_data.reshape(image_data.height, image_data.width,3)
        #print("image data : ", np.shape(in_image))
        tr_point_lc = []
        image_plane_points = []
        image_point = []
        newlist =[]
        cloud = ros_numpy.numpify(laser_data)  # direct conversion to numpy - pointcloud
        cloud = cloud.ravel() 
        cloud_in_laser = np.stack([cloud[f] for f in ['x', 'y', 'z']] + [np.ones(cloud.size)])
        #print("laser : ", np.shape(laser_data))
        #print("laser : ", laser_data)
        depth_of_image_plane = []

        # construct camera matrix
        K = np.matrix([[cam_info.K[0], cam_info.K[1], cam_info.K[2]], [cam_info.K[3], cam_info.K[4], cam_info.K[5]],
                           [cam_info.K[6], cam_info.K[7], cam_info.K[8]]])

        try:    # transform poincloud to camera frame and filter the points which are in front of camera
             transform_lc = self.tf_buffer.lookup_transform(image_data.header.frame_id, laser_data.header.frame_id,laser_data.header.stamp, timeout=rospy.Duration(2.0)) # get transformation from laser frame to camera frame
             T = ros_numpy.numpify(transform_lc.transform) # transformation matrix
             for x in range(cloud.size): #loop entire point cloud
                 point_tranformed_to_camera_plane = T.dot(cloud_in_laser[:, x]) #transform point to camera plane
                 #point_transformed_to_camera_plane = T.dot(cloud_in_laser)
                 # filter points that are behind camera
                 if point_tranformed_to_camera_plane[2] > 0.0 : tr_point_lc.append(point_tranformed_to_camera_plane) # filter  z > 1
             #print ("z :",point_tranformed_to_camera_plane[2]," all po :",point_tranformed_to_camera_plane)

        except:
             print("Laser-camera tf is not available at the moment")


        #print("image width ",image_data.width) 
        pnt = np.array(np.transpose(tr_point_lc))
        #print("shape of  my list :",pnt.shape,pnt)
        #print("len of pnt :" , len(pnt[0]))
        #this try-except aims to tranform points from camera to image plane 
        try :
             #print( "len of pnt : ",len(pnt[0]))
             ind = len(pnt[0])
             for x in range (ind):
                 Px = pnt[:3, x]
                 #print("Px hape :",np.shape(pnt[:3,x]))
                 #Px = np.array([pnt[0][x], pnt[1][x], pnt[2][x]])
                 P = Px.reshape((3,1))
                 #print("P :", P)
                 cam2image = K.dot(P)    # transform points from camera frame in image plane
                 lamba = cam2image[2]  #lambda = last row of cam2image
                 dep = math.sqrt((P[0][0]**2) + (P[1][0]**2) + (P[2][0]**2))# depth (the z coordinates of poinclouds in Camera frame)
                 #print("shpae of p :",P.shape)
                 #print("dep  :", dep)

                 ux = int(cam2image[0] / lamba) # ux
                 uy = int(cam2image[1] / lamba)  #uy
                 pixelcoord =np.array([ux,uy,dep])

                 pixels = pixelcoord.reshape(3)
                 #print("pixels :",pixels)
                 # filter out the points which are not in field of view of camera
                 if (image_data.width > pixels[0] > 0.0 and image_data.height > pixels[1] > 0.0 ) : image_plane_points.append(pixels)
                 
             #print("image data :", image_plane_points)
        except :
             print( "Camera-image tf not working ")


	         # fill the sparse depth image with the laser measurement

        image_point = np.array(np.transpose(image_plane_points))
        #print("first po :",image_point.shape)
        depth = np.empty(in_image.shape[0:2])
        depth[:] = np.nan
        try:
            ind2 = len(image_point[1])
            #print("ind2 :",ind2)

            for x in range(ind2):
                #print("point :",int(image_point[1][x]),int(image_point[0][x]),)
                depth[int(image_point[1][x])][int(image_point[0][x])] = image_point[2][x]
                #print("pixel coord dep :",int(image_point[1][x]),int(image_point[0][x]),image_point[2][x])
                #print("heyyo :",  depth[int(image_point[1][x])][int(image_point[0][x])])
            #print("depth", np.shape(depth))
        except:
            print("can't get depth")


        
        time1 = time.time()
        # evaluate network
        out_heat = {}
        scales = [1,1.5,2,4,6]
        s_i = 0
        
        for scale in scales:
            # resize image
            im = PILimage.fromarray(in_image)
            image_r = im.resize((int(im.size[0] / scale), int(im.size[1] / scale)))
            image_r = np.array(image_r)
            # transform numpy to tensor
            image_r = to_tensor(image_r)
            # evaluate model
            output = self.model(image_r)

            out_heat[s_i] = output[0, 0, :, :].detach().cpu().numpy()
            s_i += 1

        # maximum output of all scales
        max_idx = np.argmax([out_heat[0].max(), out_heat[1].max(),out_heat[2].max(), out_heat[3].max(),out_heat[4].max()])
        max_val = np.max([out_heat[0].max(), out_heat[1].max(),out_heat[2].max(), out_heat[3].max(),out_heat[4].max()])
	
        if max_val > self.det_treshhold:
            print "detected"
            det = True
            out_max = utils.max_filter(out_heat[max_idx], size=500)
            # get bbox of detection
            bbox = utils.bbox_in_image(
                np.zeros([int(in_image.shape[0] / scales[max_idx]), int(in_image.shape[1] / scales[max_idx])]), out_max,
                [32, 24], self.det_treshhold)
        
        time2 = time.time()
        #print ('detection time')
        #print (time2-time1)


        if publish_image:
            # this will draw bbox in image
            if det:
                in_image[int(bbox[0, 1] * image_data.height):int(bbox[0, 3] * image_data.height),
                int(bbox[0, 0] * image_data.width):int(bbox[0, 0] * image_data.width) + 2, 1] = 255
                in_image[int(bbox[0, 1] * image_data.height):int(bbox[0, 3] * image_data.height),
                int(bbox[0, 2] * image_data.width) - 3:int(bbox[0, 2] * image_data.width) - 1, 1] = 255
                in_image[int(bbox[0, 1] * image_data.height):int(bbox[0, 1] * image_data.height) + 2,
                int(bbox[0, 0] * image_data.width):int(bbox[0, 2] * image_data.width), 1] = 255
                in_image[int(bbox[0, 3] * image_data.height) - 3:int(bbox[0, 3] * image_data.height) - 1,
                int(bbox[0, 0] * image_data.width):int(bbox[0, 2] * image_data.width), 1] = 255

            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.data = in_image.tostring()
            msg.height = image_data.height
            msg.width = image_data.width
            msg.step = 1920
            msg.encoding = 'rgb8'
            msg.is_bigendian = 0
            self.image_pub.publish(msg)


	# if barbie detected
        if det:
            # create mask for detected bbox
            mask = np.zeros(in_image.shape[0:2]).astype(np.uint8)
            mask[int(bbox[0, 1] * image_data.height):int(bbox[0, 3] * image_data.height),
            int(bbox[0, 0] * image_data.width):int(bbox[0, 2] * image_data.width)] = 1

            # estimate the depth
            depths = depth[mask == 1]
            depths = depths[~np.isnan(depths)]
            if (len(depths)>0) & (sum(depths)>0):
                d = np.percentile(depths[depths != 0], 50)
            else:
                print "no depth"
                # if no depth information known -> skip
                return
            # (u,v) coordinates of the detected barbie
            u = np.argmax(out_max, axis=1).max()*8*scales[max_idx]
            v = np.argmax(out_max, axis=0).max()*8*scales[max_idx]

            # TODO

            # project point to X,Y,Z in camera frame
            """
            Kinv = np.linalg.inv(K)
            U = np.matrix([[u], [v], [1]])
            B = d * Kinv * U """
                             
            try :
                Kinv = np.linalg.inv(K)  #here I aplied the opposite of camera to image plane
                U = np.matrix([[u], [v], [1]])
                #print("u :", U,np.shape(Kinv),np.shape(U))
                B =  d* (Kinv.dot(U))
                #print("B sahpe :", np.shape(B))
                coord = np.array([B[0][0],B[1][0], B[2][0],np.ones(1)]) #this is create vecotr with 4 elements for dot product
                cord = coord.reshape(4,1) #reshape for multiplication
                #print("coor shape :", np.shape(cord),cord)
                #print("d :",d )
                transform2 = self.tf_buffer.lookup_transform("map", image_data.header.frame_id, image_data.header.stamp, timeout=rospy.Duration(2.0))
                
                T2 = ros_numpy.numpify(transform2.transform)
                print("t2 :", T2)
                print("point_in_map", T2.dot(cord))
                point_in_map = T2.dot(cord)
                print("last point :",point_in_map)
                #print("last x:",point_in_map[0][0] )
                barbie_point = PointStamped()
                barbie_point.header = image_data.header
                barbie_point.header.frame_id = "map"
                barbie_point.point.x = (point_in_map[0][0])
                barbie_point.point.y = (point_in_map[1][0])
                barbie_point.point.z = (point_in_map[2][0])
                self.point_pub.publish(barbie_point)
                self.barbie_status = "FOUND"

            except :
                print("Second tranformation not working ")      

            # transform point in map frame
            # point_in_map = ...
            #status = self.barbie_status
            #self.barbie_status_pub.publish(status)
            """
            # create a 3D point
            barbie_point = PointStamped()
            barbie_point.header = image_data.header
            barbie_point.header.frame_id = "map"
            barbie_point.point.x = point_in_map[0]
            barbie_point.point.y = point_in_map[1]
            barbie_point.point.z = point_in_map[2]
            self.point_pub.publish(barbie_point)
                                                """


if __name__ == '__main__':
    ic = barbie_detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
