import cv2
import numpy as np


MAP_SCALE = 4
MAP_SIZE = 400
MAP_UNKNOWN_COLOR = np.r_[142, 135, 67].astype(float) / 255


def convertMap(mapImage, scale=MAP_SCALE):
    image = 1 - (mapImage).astype(float) / 100
    image = np.tile(image[..., np.newaxis], (1, 1, 3))
    unknown = np.where(mapImage < 0)
    image[unknown[0], unknown[1], :] = MAP_UNKNOWN_COLOR
    image = np.flipud(image)
    # return cv2.resize(image, tuple((np.r_[mapImage.shape] * MAP_SCALE)), interpolation=cv2.INTER_NEAREST)

    shape = np.r_[mapImage.shape]
    return cv2.resize(image, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_NEAREST)


def showMap(mapImage, title="map", scale=MAP_SCALE):
    image = convertMap(mapImage)
    cv2.imshow(title, image)
    return cv2.waitKey(1)
