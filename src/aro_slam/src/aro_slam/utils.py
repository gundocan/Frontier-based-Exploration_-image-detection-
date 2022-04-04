from __future__ import absolute_import, division, print_function
import numpy as np
from ros_numpy import msgify, numpify
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))


def col(arr):
    """Convert array to column vector."""
    assert isinstance(arr, np.ndarray)
    return arr.reshape((arr.size, 1))


def logistic(x):
    """Standard logistic function, inverse of logit function."""
    return 1. / (1. + np.exp(-x))


def logit(p):
    """Logit function or the log-odds, inverse of logistic function.
    The logarithm of the odds p / (1 - p) where p is probability."""
    return np.log(p / (1. - p))


def affine_transform(T, x):
    """Affine transform T * [x; 1]. The last row in T is neglected."""
    assert isinstance(T, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert T.ndim >= 2
    assert x.ndim >= 2
    d = T.shape[-1] - 1
    # d = x.shape[0]
    # assert d <= T.shape[0] <= d + 1
    # assert T.shape[1] == d + 1
    assert d <= T.shape[-2] <= d + 1
    assert d <= x.shape[-2] <= d + 1
    R = T[..., :d, :d]
    t = T[..., :d, d:]
    return np.matmul(R, x) + t


def inverse_affine(T):
    """Invert affine transform, [R t]^-1 = [R^T -R^T*t]."""
    assert isinstance(T, np.ndarray)
    assert T.ndim >= 2
    d = T.shape[-1] - 1
    assert d <= T.shape[-2] <= d + 1
    R = T[:d, :d]
    t = T[:d, d:]
    T_inv = T.copy()
    # T_inv = np.eye(d + 1)
    T_inv[..., :d, :d] = R.T
    T_inv[..., :d, d:] = -np.matmul(R.T, t)
    return T_inv


def rotation_angle(R):
    """Rotation angle from a rotation matrix."""
    assert isinstance(R, np.ndarray)
    assert R.ndim >= 2
    assert R.shape[-2] == R.shape[-1]
    d = R.shape[-1]
    alpha = np.arccos((np.trace(R, axis1=-2, axis2=-1) - (d - 2)) / 2.)
    return alpha


def msg_to_cloud(msg, fields=('x', 'y', 'z')):
    """Convert PointCloud2 message to F-by-N array using given F fields."""
    x = numpify(msg).ravel()
    x = np.stack([x[f] for f in fields])
    return x


def cloud_to_msg(x, fields=('x', 'y', 'z')):
    """Convert cloud F-by-N array to PointCloud2 message."""
    dtype = list(zip(fields, len(fields) * [np.float32]))
    cloud = np.zeros(x.shape[1], dtype=dtype)
    for i, f in enumerate(fields):
        cloud[f] = x[i, :]
    msg = msgify(PointCloud2, cloud)
    return msg


def filter_grid(x, grid_res, keep='random'):
    """Select random point within each cell. Order is not preserved."""
    # Convert points in cols to rows, and shuffle the rows.
    if keep == 'first':
        # Make the first item last.
        x = x[:, ::-1].T
    elif keep == 'random':
        # Make the last item random.
        x = x.T.copy()
        np.random.shuffle(x)
    elif keep == 'last':
        # Keep the last item last.
        pass
    # Get integer cell indices, as tuples.
    idx = np.floor(x / grid_res)
    idx = [tuple(i) for i in idx]
    # Dict keeps the last value for each key, which set above by keep param.
    x = dict(zip(idx, x))
    # Concatenate and convert rows to cols.
    x = np.stack(x.values()).T
    return x


def zeropad(x, shape):
    assert np.all(np.less_equal(x.shape, shape))
    pad = [(0, n - m) for m, n in zip(x.shape, shape)]
    x = np.pad(x, pad, 'constant')
    return x


class PointMap(object):
    def __init__(self, grid_res=None, max_size=None):
        self.cloud = None
        self.index = None
        self.grid_res = grid_res
        self.max_size = max_size

    def size(self):
        if self.cloud is None:
            return 0
        return self.cloud.shape[1]

    def empty(self):
        return self.size() == 0

    def update(self, x):
        if self.empty():
            self.cloud = x
        else:
            self.cloud = np.concatenate((self.cloud, x), axis=1)
        if self.grid_res is not None:
            self.cloud = filter_grid(self.cloud, self.grid_res, keep='first')
        if self.max_size is not None and self.size() > self.max_size:
            # Keep oldest points.
            # self.cloud = self.cloud[:, :self.max_size]
            # Keep most recent points.
            self.cloud = self.cloud[:, -self.max_size:]

    def update_index(self):
        self.index = cKDTree(self.cloud.T)
