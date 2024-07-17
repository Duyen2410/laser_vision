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

