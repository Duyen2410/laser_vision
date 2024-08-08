import numpy as np
import time
from sklearn.metrics import mean_squared_error

def hough_transform_3d(points, theta_res=0.1, phi_res=0.1, rho_res=0.1):
    points = np.asarray(points)
    
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]

    # Define the ranges for theta, phi, and d
    theta_max = np.pi
    phi_max = 2 * np.pi
    rho_max = np.linalg.norm(points, axis=1).max()
    
    # Create the accumulator array
    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    rho_bins = int((2*rho_max) / rho_res)
    accumulator = np.zeros((theta_bins, phi_bins, rho_bins))


    for x, y, z in points:
        for theta_idx in range(theta_bins):
            theta = theta_idx * theta_res
            for phi_idx in range(phi_bins):
                phi = phi_idx * phi_res
                a = np.sin(theta) * np.cos(phi)
                b = np.sin(theta) * np.sin(phi)
                c = np.cos(theta)
                rho = a * x + b * y + c * z
                rho_idx = int((rho + rho_max)/theta_res)
                accumulator[theta_idx, phi_idx, rho_idx] += 1

    idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    theta_idx_f, phi_idx_f, rho_idx_f = idx
    best_theta = theta_idx_f * theta_res
    best_phi = phi_idx_f * phi_res
    best_rho = rho_idx_f * rho_res - rho_max

    a = np.sin(best_theta) * np.cos(best_phi)
    b = np.sin(best_theta) * np.sin(best_phi)
    c = np.cos(best_theta)

    best_fit_plane = (a, b, c, -best_rho)
    return xs, ys, zs, best_fit_plane

def evaluate_models(points, best_plane_ransac):
    X = points[:, :2]
    true_z = points[:, 2]
    
    # Dự đoán z từ mô hình RANSAC
    a_r, b_r, c_r, d_r = best_plane_ransac
    pred_z_ransac = -(a_r * X[:, 0] + b_r * X[:, 1] + d_r)/c_r
    mse_ransac = mean_squared_error(true_z, pred_z_ransac)
    sse_ransac = np.sum((true_z - pred_z_ransac) ** 2)
    sst = np.sum((true_z - np.mean(true_z)) ** 2)
    # Tính R^2
    r2_ransac = 1 - (sse_ransac / sst)
    n = len(points)
    p = 2  # Số biến độc lập
    r2_adj_ransac = 1 - ((1 - r2_ransac) * (n - 1)) / (n - p - 1)
    mae = np.mean(np.abs(true_z - pred_z_ransac))

    return mse_ransac, r2_adj_ransac, mae



