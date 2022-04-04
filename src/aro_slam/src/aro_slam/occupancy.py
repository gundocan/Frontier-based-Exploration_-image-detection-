from __future__ import absolute_import, division, print_function
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import numpy as np
from .utils import array, col
from voxel_map import VoxelMap


def logistic(x):
    """Standard logistic function, inverse of logit function."""
    return 1. / (1. + np.exp(-x))


def logit(p):
    """Logit function or the log-odds, inverse of logistic function.
    The logarithm of the odds p / (1 - p) where p is probability."""
    return np.log(p / (1. - p))


class OccupancyMap(object):
    def __init__(self, frame_id, resolution=0.1, empty_update=-1.0, occupied_update=1.0,
                 min=-10.0, max=10.0, occupied=5.0):
        # self.voxel_map = VoxelMap(resolution, -1.0, 1.0, occupied)
        self.voxel_map = VoxelMap(resolution, empty_update, occupied_update, occupied)
        self.min = min
        self.max = max
        self.msg = OccupancyGrid()
        self.msg.header.frame_id = frame_id
        self.msg.info.resolution = resolution
        # Initially, set the grid origin to identity.
        self.msg.info.origin.orientation.w = 1.0

    def map_to_grid(self, x):
        """Transform points from map coordinates to grid."""
        # TODO: Handle orientation too.
        x = x - col(array(self.msg.info.origin.position))
        return x

    def grid_to_map(self, x):
        """Transform points from grid coordinates to map."""
        # TODO: Handle orientation too.
        x = x + col(array(self.msg.info.origin.position))
        return x

    def fit_grid(self):
        """Accommodate the grid to contain all points."""
        # Update grid origin so that all coordinates are non-negative.
        x, _, v = self.voxel_map.get_voxels()
        if x.size == 0:
            return
        x = x[:2]  # Only x,y used in 2D grid.
        x_min = x.min(axis=1) - self.voxel_map.voxel_size / 2.
        x_max = x.max(axis=1) + self.voxel_map.voxel_size / 2.
        nx = np.round((x_max - x_min) / self.msg.info.resolution).astype(np.int)
        self.msg.info.origin.position = Point(x_min[0], x_min[1], 0.0)
        self.msg.info.width, self.msg.info.height = nx

    def grid_voxels(self):
        """Return voxel coordinates corresponding to the current grid."""
        i, j = np.meshgrid(np.arange(self.msg.info.width),
                           np.arange(self.msg.info.height),
                           indexing='xy')
        x = np.stack((i.ravel(), j.ravel(), np.zeros_like(i).ravel()))
        x = (x + 0.5) * self.msg.info.resolution
        return x

    def to_msg(self):
        """Return as occupancy grid message. Update grid parameters as needed."""
        self.fit_grid()
        x = self.grid_voxels()
        x = self.grid_to_map(x)
        x[2, :] = self.voxel_map.voxel_size / 2.0
        l = np.zeros((x.shape[1],))
        v = self.voxel_map.get_voxels(x, l)
        v = 100. * logistic(v)
        v[np.isnan(v)] = -1.
        self.msg.data = v.astype(int).tolist()
        return self.msg
    
    def voxel_map_points(self, x):
        x = x.copy()
        x[2, :] = self.voxel_map.voxel_size / 2.0
        return x

    def update(self, x, y, stamp):
        """Update internal occupancy map."""
        x = self.voxel_map_points(x)
        y = self.voxel_map_points(y)
        if x.shape[1] == 1:
            x = np.broadcast_to(x, y.shape)
        elif y.shape[1] == 1:
            y = np.broadcast_to(y, x.shape)
        self.voxel_map.update_lines(x, y)
        self.clip_values()
        self.msg.header.stamp = stamp
        self.msg.info.map_load_time = stamp
    
    def occupied(self, x):
        """Occupied flags for points x."""
        x = self.voxel_map_points(x)
        l = np.zeros((x.shape[1],))
        v = self.voxel_map.get_voxels(x, l)
        occupied = v > self.voxel_map.occupied_threshold
        return occupied
    
    def clip_values(self):
        """Clip values between min and max."""
        x, l, v = self.voxel_map.get_voxels()
        v = np.clip(v, self.min, self.max)
        self.voxel_map.set_voxels(x, l, v)
