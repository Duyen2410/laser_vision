import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import least_squares

def calculate_plane_parameters(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = np.dot(normal, np.array(p1))
    rho = d / np.linalg.norm(normal)
    theta = np.arctan2(b, a)
    phi = np.arccos(c / np.linalg.norm(normal))
    return theta, phi, rho

def increment_accumulator(theta, phi, rho, theta_bins, phi_bins, rho_bins, max_rho):
    theta_idx = int(theta * theta_bins / np.pi)
    phi_idx = int(phi * phi_bins / (2 * np.pi))
    rho_idx = int((rho + max_rho) * rho_bins / (2 * max_rho))
    if 0 <= theta_idx < theta_bins and 0 <= phi_idx < phi_bins and 0 <= rho_idx < rho_bins:
        return theta_idx, phi_idx, rho_idx
    return None

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_close_to_plane(point, theta, phi, rho, tolerance=0.1):
    x, y, z = point
    return abs(x * np.cos(theta) * np.sin(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(phi) - rho) < tolerance

def most_common(lst):
    return max(set(lst), key=lst.count)

def p_hough_transform_3d(points, theta_res=0.05, phi_res=0.05, rho_res=0.05, threshold_t=10):
    B = []

    points = np.asarray(points)
    theta_max = np.pi
    phi_max = 2 * np.pi
    rho_max = np.linalg.norm(points, axis=1).max()

    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    rho_bins = int((2 * rho_max) / rho_res)
    A = np.zeros((theta_bins, phi_bins, rho_bins))

    while len(points) > 0:
        p1, p2, p3 = points[np.random.choice(points.shape[0], 3, replace=True)]
        if distance(p1, p2) > 1e-3 and distance(p1, p3) > 1e-3 and distance(p2, p3) > 1e-3:
            theta, phi, rho = calculate_plane_parameters(p1, p2, p3)
            cell = increment_accumulator(theta, phi, rho, theta_bins, phi_bins, rho_bins, rho_max)

            if cell:
                theta_idx, phi_idx, rho_idx = cell
                A[theta_idx, phi_idx, rho_idx] += 1

                if A[theta_idx, phi_idx, rho_idx] >= threshold_t:
                    B.append(cell)
                    mask = np.apply_along_axis(is_close_to_plane, 1, points, theta, phi, rho)
                    points = np.delete(points, np.where(mask), axis=0)
                    A = np.zeros((theta_bins, phi_bins, rho_bins))


    if len(B) == 0:
        raise ValueError("No plane was found.")

    r = most_common(B)
    theta_idx_f, phi_idx_f, rho_idx_f = r
    best_theta = theta_idx_f * theta_res
    best_phi = phi_idx_f * phi_res
    best_rho = rho_idx_f * rho_res - rho_max

    a = np.sin(best_theta) * np.cos(best_phi)
    b = np.sin(best_theta) * np.sin(best_phi)
    c = np.cos(best_theta)
    best_fit_plane = (a, b, c, -best_rho)

    def plane_func(params, points):
        a, b, c, d = params
        return a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

    initial_params = [a, b, c, -best_rho]
    res_lsq = least_squares(plane_func, initial_params, args=(points,))
    refined_plane = res_lsq.x

    return refined_plane

np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))

# Hough transform
start_time = time.time()
best_fit = p_hough_transform_3d(points)
hough_time = time.time() - start_time

a_h, b_h, c_h, d_h = float(best_fit[0]), float(best_fit[1]), float(best_fit[2]), float(best_fit[3])

def evaluate_models(points, best_plane_hough):
    X = points[:, :2]
    true_z = points[:, 2]
    a_h, b_h, c_h, d_h = best_plane_hough
    pred_z_hough = -(a_h * X[:, 0] + b_h * X[:, 1] + d_h) / c_h
    mse_hough = mean_squared_error(true_z, pred_z_hough)
    return mse_hough

mse_hough = evaluate_models(points, best_fit)

print("\n=== So sánh các phương pháp ===")
print(f"Phương trình mặt phẳng từ Hough: z = {-a_h/c_h:.4f} * x + {-b_h/c_h:.4f} * y + {-d_h/c_h:.4f}")
print(f"Thời gian thực thi với Hough: {hough_time:.4f} giây")
print(f"MSE của Hough transform: {mse_hough:.4f}")
