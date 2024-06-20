'''Different error fuctions for plane fitting

'''
from typing import Tuple, List, Callable

import numpy as np
import open3d as o3d

def ransac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    
    inliers = distances - threshold
    error = np.sum(~inliers)

    return error, inliers

def mlesac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    
    sigma = threshold/2
    v = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    gamma = 1/2
    for j in range(3):

        p_i = gamma*1/(np.sqrt(2*np.pi)*sigma)*np.exp(-pow(distances,2)/(2*pow(sigma,2)))
        p_o = (1-gamma)/v
        z_i = p_i/(p_i+p_o)
        gamma = np.sum(z_i)/len(distances)
        error = -np.sum(np.log(p_i+p_o))
        inliers = distances < threshold

        return error, inliers

def fit_plane(pcd: o3d.geomatry.PointCloud,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    

    points = np.asarray(pcd.points)
    N = len(points)
    m = 3
    eta_0 = 1-confidence
    k, eps, error_star = 0, m/N, np.inf
    I = 0
    best_inliers = np.full(shape=(N,), fill_value=0.)
    best_plane = np.full(shape=(4,), fill_value=-1.)
    while pow((1-pow(eps, m)), k) >= eta_0:
        p1, p2, p3 = points[np.random.randint(N)], points[np.random.randint(N)], 
                    points[np.random.randint(N)]
        if np.linalg.norm(p1-p2) < min_sample_distance or np.linalg.norm(p2-p3) < min_sample_distance or np.linalg.norm(p1-p3) < min_sample_distance: 
            continue
        n = np.cross(p2-p1, p3-p1)
        n = n/np.linalg.norm(n)

        if n[2] < 0:
             n = -n
        d = -np.dot(n,p1)
        distances = np.abs(np.dot(points, n)+d)
        error, inliers = error_func(pcd, distances, inlier_threshold)
        if error < error_star:
             I = np.sum(inliers)
             eps = I/N
             best_inliers = inliers
             error_star = error
        k = k + 1
        A = points[best_inliers]
        y = np.full(shape=(len(A),), fill_value=1.)
        best_plane[0:3] = np.linalg.lstsq(A,y,rcond=-1)[0]
        if best_plane[2] < 0:
             best_plane = -best_plane
        return best_plane, best_inliers, k

def filter_planes(pcd: o3d.geometry.PointCloud,
                  min_points_prop: float,
                  confidence : float,
                  inlier_threshold: float,
                  error_func: Callable) -> Tuple[List[np.ndarray],
                                                 List[o3d.geometry.PointCloud],
                                                 o3d.geometry.PointCloud]:
    filtered_pcd = copy.deepcopy(pcd)
    filtered_points = np.asarray(filtered_pcd.points)
    N = len(filtered_points)
    plane_eqs = []
    plane_pcds = []
    while True:
        best_plane, best_inliers, num_iterations = fit_plane(filtered_pcd, confidence, inlier_threshold, min_sample_distance, error_func)
        if np.sum(best_inliers)/N <= min_points_prop:
            break
        plane_eqs.append(best_plane)
        plane_pcds.append(filtered_pcd.select_by_index(np.nonzero(best_inliers)[0]))
        filtered_pcd = filtered_pcd.select_by_index(np.nonzero(best_inliers)[0], invert=True)

    return plane_eqs, plane_pcds, filtered_pcd



