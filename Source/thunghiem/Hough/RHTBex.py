import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

def detect_best_plane(D, k_max_in = 100, n_in = 0.1, theta_res=0.05, phi_res=0.05, rho_res=0.05):
   
    points = np.asarray(D)
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]

    k_max = int(k_max_in*len(points))
    n = int(n_in*len(points))

    theta_max = 2*np.pi
    phi_max = np.pi
    rho_max = np.linalg.norm(points, axis=1).max()

    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    rho_bins = int(rho_max / rho_res)
    A = np.zeros((theta_bins, phi_bins, rho_bins))


    def select_random(points, n):
        indices = random.sample(range(points.shape[0]), n)
        return points[indices]

    def fit(points):
        xs = points[:,0]
        ys = points[:,1]
        zs = points[:,2]
        tmp_A = np.c_[xs, ys, zs, np.ones(xs.shape[0])]
        U, S, Vt = np.linalg.svd(tmp_A)
        model = Vt[-1, :] # Giải pháp là hàng cuối cùng của Vt
        a, b, c, d = model
        rho = -d / np.linalg.norm(model[:3])
        phi = np.arccos(c / np.linalg.norm(model[:3]))  # Calculate theta using arccos
        theta = np.arctan2(b, a)  # Calculate phi using arctan2
        return theta, phi, rho

    db = DBSCAN(eps=0.5, min_samples=5).fit(points)
    labels = db.labels_
    sorted_indices = np.argsort(labels)  # Sắp xếp theo nhãn
    points_sorted = points[sorted_indices]
    i = 0
    
    for k in range(k_max):
        subset_indices = random.sample(range(min(i + n, len(points))), n)
        i += 5
        sample = points_sorted[subset_indices]
        #sample = select_random(points, n)
        model = fit(sample)
        theta, phi, rho = model
        theta_idx = int(theta / theta_res)
        phi_idx = int(phi / phi_res)
        rho_idx = int((rho + rho_max) / rho_res)
        if 0 <= rho_idx < rho_bins :
            A[theta_idx, phi_idx, rho_idx] += 1

      
    idx = np.unravel_index(np.argmax(A), A.shape)

    theta_idx, phi_idx, rho_idx = idx
    best_theta = theta_idx * theta_res
    best_phi = phi_idx * phi_res
    best_rho = rho_idx * rho_res - rho_max
    a = np.sin(best_phi) * np.cos(best_theta)
    b = np.sin(best_phi) * np.sin(best_theta)
    c = np.cos(best_phi)
    best_fit_planes = (a, b, c, -best_rho)

    return xs, ys, zs, best_fit_planes



def evaluate_models(points, best_plane_hough):
    X = points[:, :2]
    true_z = points[:, 2]
    
    
    # Dự đoán z từ mô hình Hough
    a_h, b_h, c_h, d_h = best_plane_hough
    pred_z_hough = -(a_h * X[:, 0] + b_h * X[:, 1] + d_h)/c_h
    
    # Tính MSE cho từng mô hình

    mse_hough = mean_squared_error(true_z, pred_z_hough)
    
    return mse_hough





