from __future__ import absolute_import, division, print_function
import numpy as np
import rospy
from scipy.spatial import cKDTree
from .utils import affine_transform


def absolute_orientation(x, y):
    """Find transform R, t between x and y, such that the sum of squared
    distances ||R * x[:, i] + t - y[:, i]|| is minimum.

    :param x: Points to align, D-by-M array.
    :param y: Reference points to align to, D-by-M array.

    :return: Optimized transform from SE(D) as (D+1)-by-(D+1) array,
        T = [R t; 0... 1].
    """
    assert x.shape == y.shape
    d = x.shape[0]
    T = np.eye(d + 1)
    
  
   # if R is None :
    R = np.eye(d)
    #if t is None :
    t=np.zeros((d,1))

    dummy_array= []
    dummy_array = np.array([[0, 0, 1]])
 
    #center_points_x =np.mean(x,0)    
    #center_points_y=np.mean(y,0)

    x1p = np.mean(x[0], dtype=np.float64) #take mean of given x and y list
    x2p = np.mean(x[1], dtype=np.float64)
    y1p = np.mean(y[0], dtype=np.float64)
    y2p = np.mean(y[1], dtype=np.float64)

      
    x-= np.array((np.ones(len(x[0]),)*x1p, np.ones(len(x[0]),)*x2p), np.float64) # x- p_hat
    y-= np.array((np.ones(len(y[0]),)*y1p, np.ones(len(y[0]),)*y2p), np.float64)  #y - q_hat

    #p_hat =np.array(np.tile(center_points_x,(d,1)))
   # q_hat =np.array(np.tile(center_points_y,(d,1)))

    #new_points_x =np.array (x - p_hat)
    #new_points_y = np.array(y - q_hat)

    H = x.dot(y.T)   
    U,S,V = np.linalg.svd(H, full_matrices=True)
    V = V.T
    R = V.dot(U.T)
    #t=np.array(q_hat - R.dot(p_hat))
    t = np.array((y1p, y2p), np.float64) - R.dot(np.array((x1p, x2p), np.float64))
    t = np.array((np.ones(1,)*t[0], np.ones(1,)*t[1]), np.float64)
    

    


    for x in range(d): #this loop is just used for building T matrix
        #print("dummy array :",dummy_array)
            
        T=np.concatenate((R, t), axis=1)
        T=np.concatenate((T,dummy_array))

    return T


def icp(x, y, y_index=None, max_iters=50, inlier_ratio=1.0, inlier_dist_mult=1.0, T=None, R=None,t=None):
    """Iterative closest point algorithm, minimizing sum of squares of point
    to point distances.

    :param x: Points to align, D-by-M array.
    :param y: Reference points to align to, D-by-N array, such that
            [y[:, j]; 1] is approximately T*[x[:, i]; 1] for corresponding
            pair (i, j). The correspondences are established from nearest
            neighbors (i.e., "closest points").
    :param y_index: Index for NN search for y (cKDTree)
            which can be queried with row 1-by-D vectors.
    :param max_iters: Maximum number of iterations.
    :param inlier_ratio: Ratio of inlier correspondences with lowest
            nearest-neighbor distances for which we optimize the criterion
            in given iteration. This should correspond to the minimum relative
            overlap between point clouds. The inliers set may change each
            iteration.
    :param inlier_dist_mult: Multiplier of the maximum inlier distance found
            using inlier ratio above, enlarging or reducing the inlier set for
            optimization.
    :param T: Initial transform estimate from SE(D), defaults to identity.
    :return: Optimized transform from SE(D) as (D+1)-by-(D+1) array,
            mean inlier error (L2 distance) from the last iteration, and
            boolean inlier mask from the last iteration, x[:, inl] are the
            inlier points.
    """
    #print ("X :", x )
    #print("Y : ",y) 
    assert x.shape[0] == y.shape[0]
    #xs=x.shape
    #ys=y.shape 
    #print("xs : ",xs )
    #print("ys :  ",ys)
    d = x.shape[0]
    #print("x_shp_0 :",d)
    
    if y_index is None:
        y_index = cKDTree(y.T)
        
        
    if T is None:
        T = np.eye(d + 1)
    if R is None:
        R = np.eye(x.shape[0])
    if t is None:
        t = np.zeros((x.shape[0], 1))
        
    # Boolean inlier mask from current iteration.
    inl = None
    # Mean inlier distance from current iteration.
    inl_err = float("inf")
    # Mean inlier distance from previous iteration (to assess improvement).
    prev_inl_err = float("inf")
    # TODO: Implement!
    #print("y index : ", str(y_index)) 

    Ru = R
    tu = t
    x = R.dot(x) + t
    d_a =np.array([[0, 0, 1]])  
     
    for z in range(5):  #iterate the loop 
        nbrs , ind = y_index.query(x.T)
            #print("neighbors :", nbrs)
            #print("ind :", ind)    
        #prev_inl_err = np.mean(nbrs, dtype=np.float64) #Previous mean inlier distance

        outlier = (inlier_dist_mult*np.percentile(nbrs, inlier_ratio*100)) #define outlier
        inl = nbrs <= outlier   #reject median threshold
        inl_err = np.mean(nbrs[inl] , dtype=np.float64)    #Current mean inlier distance nbrs[inl]
            
        x_tmp = x[:, inl]
        ind = ind[inl]
        y_tmp = y[:, ind]
        Tr = absolute_orientation(x_tmp,y_tmp)
        T=Tr
        R=np.array([[T[0][0] ,T[0][1]], [T[1][0] ,T[1][1]]])
        t=np.array([[T[0][2]], [T[1][2]]])
            
        x = R.dot(x) + t

        Ru = R.dot(Ru)
        tu = R.dot(tu) + t
            #print("tu :" ,tu)
            #print("RU :",Ru )
            
        T=np.concatenate((Ru, tu), axis=1)
        T=np.concatenate((T,d_a))
    
    return T,inl_err, inl
