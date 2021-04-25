import numpy as np
import tf.transformations as tft


def getRobotGridPosition(transMsg, gridInfo):
    pos = np.array([transMsg.transform.translation.x - gridInfo.origin.position.x, transMsg.transform.translation.y - gridInfo.origin.position.y, 0, 1])
    quat = gridInfo.origin.orientation
    mat = tft.quaternion_matrix(tft.quaternion_inverse([quat.x, quat.y, quat.z, quat.w]))
    gridPos = (mat.dot(pos[np.newaxis].T).flatten()[:2]) / gridInfo.resolution
    roundedPos = np.round(gridPos)
    pos = roundedPos if np.allclose(gridPos, roundedPos) else np.floor(gridPos)
    return pos
    
def getGridPosition(pose2d, gridInfo):
    pos = np.array([pose2d.x - gridInfo.origin.position.x, pose2d.y - gridInfo.origin.position.y, 0, 1])
    quat = gridInfo.origin.orientation
    mat = tft.quaternion_matrix(tft.quaternion_inverse([quat.x, quat.y, quat.z, quat.w]))
    gridPos = (mat.dot(pos[np.newaxis].T).flatten()[:2]) / gridInfo.resolution
    roundedPos = np.round(gridPos)
    pos = roundedPos if np.allclose(gridPos, roundedPos) else np.floor(gridPos)
    return pos

def mapToGridCoordinates(position, gridInfo):
    pos = np.array([position[0] - gridInfo.origin.position.x, position[1] - gridInfo.origin.position.y, 0, 1])
    quat = gridInfo.origin.orientation
    mat = tft.quaternion_matrix(tft.quaternion_inverse([quat.x, quat.y, quat.z, quat.w]))
    gridPos = (mat.dot(pos[np.newaxis].T).flatten()[:2]) / gridInfo.resolution
    roundedPos = np.round(gridPos)
    pos = roundedPos if np.allclose(gridPos, roundedPos) else np.floor(gridPos)
    return pos

def gridToMapCoordinates(position, gridInfo):
    position = position * gridInfo.resolution
    originPos = np.array([gridInfo.origin.position.x, gridInfo.origin.position.y])
    pos = np.array([position[0], position[1], 0, 1])
    quat = gridInfo.origin.orientation
    mat = tft.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    pos = mat.dot(pos[np.newaxis].T).flatten()[:2] + originPos
    return pos
