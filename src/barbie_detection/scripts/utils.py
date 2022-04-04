import numpy as np
from scipy import ndimage


def bbox_image_all(in_im, out_im, bbox):
    # returns all bounding boxes in image (in_im) that corresponds to coordinates defined by out_im
    bbox_list = []
    for i in range(0,out_im.shape[0]):
        for j in range(0, out_im.shape[1]):
            position_i = int((float(i)/float(out_im.shape[0]))*in_im.shape[0])
            position_j = int((float(j)/float(out_im.shape[1]))*in_im.shape[1])
            y_a = max(position_i-bbox[0], 0)
            y_b = min(position_i+bbox[0], in_im.shape[0])
            x_a = max(position_j-bbox[1], 0)
            x_b = min(position_j+bbox[1], in_im.shape[1])
            bbox_list.append([float(x_a)/in_im.shape[1],float(y_a)/in_im.shape[0],float(x_b)/in_im.shape[1],float(y_b)/in_im.shape[0]])
    bbox_list = np.array(bbox_list).reshape((-1, 4))
    return bbox_list


def bbox_in_image(in_im, out_im, bbox, threshold):
    # returns all bounding boxes that corresponds values bigger than threshold in out_im
    bbox_list = []
    for i in range(0,out_im.shape[0]):
        for j in range(0, out_im.shape[1]):
            if out_im[i,j]>threshold:
                position_i = int((float(i)/float(out_im.shape[0]))*in_im.shape[0])
                position_j = int((float(j)/float(out_im.shape[1]))*in_im.shape[1])
                y_a = max(position_i-bbox[0], 0)
                y_b = min(position_i+bbox[0], in_im.shape[0])
                x_a = max(position_j-bbox[1], 0)
                x_b = min(position_j+bbox[1], in_im.shape[1])
                bbox_list.append([float(x_a)/in_im.shape[1],float(y_a)/in_im.shape[0],float(x_b)/in_im.shape[1],float(y_b)/in_im.shape[0], out_im[i,j]])
    bbox_list = np.array(bbox_list).reshape((-1, 5))

    return bbox_list


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    if (xB - xA)<0  or (yB - yA)< 0:
        interArea = 0
    else:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
    # compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def max_filter(label,size=20):
    # returns image with only maximum value	
    label += np.random.rand(label.shape[0],label.shape[1])/10000
    image_max = ndimage.maximum_filter(label, size=size, mode='constant')
    mask = (label == image_max)
    label_out = label*mask
    return label_out


