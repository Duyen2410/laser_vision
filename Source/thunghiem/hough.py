import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from itertools import product
import random
from sklearn.metrics import mean_squared_error
def hough_transform_3d(points, theta_res=0.1, phi_res=0.1, d_res=0.1):
    points = np.asarray(points)
    
    # Define the ranges for theta, phi, and d
    theta_max = np.pi
    phi_max = 2 * np.pi
    d_max = np.linalg.norm(points, axis=1).max()
    
    # Create the accumulator array
    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    d_bins = int((2*d_max) / d_res)
    accumulator = np.zeros((theta_bins, phi_bins, d_bins))
    
    for x, y, z in points:
        for theta_bin in range(theta_bins):
            theta = theta_bin * theta_res
            for phi_bin in range(phi_bins):
                phi = phi_bin * phi_res
                a = np.sin(theta) * np.cos(phi)
                b = np.sin(theta) * np.sin(phi)
                c = np.cos(theta)

                d = -(a * x + b * y + c * z)
                
                # Normalize (a, b, c)
                norm = np.sqrt(a**2 + b**2 + c**2)
                a /= norm
                b /= norm
                c /= norm
                d /= norm
                
                # Discretize d
                d_bin = int((d + d_max) / d_res)
                
                if 0 <= d_bin < d_bins:
                    accumulator[theta_bin, phi_bin, d_bin] += 1

    # Find the index of the maximum value in the accumulator
    max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    best_theta_bin, best_phi_bin, best_d_bin = max_idx
    best_theta = best_theta_bin * theta_res
    best_phi = best_phi_bin * phi_res
    best_d = best_d_bin * d_res - d_max
    
    # Calculate the normal vector (a, b, c)
    a = np.sin(best_theta) * np.cos(best_phi)
    b = np.sin(best_theta) * np.sin(best_phi)
    c = np.cos(best_theta)
    
    # Normalize (a, b, c)
    norm = np.sqrt(a**2 + b**2 + c**2)
    a /= norm
    b /= norm
    c /= norm
    best_d /= norm
    
    best_fit_plane = (a, b, c, best_d)
    return best_fit_plane


np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))

# Phương pháp Hough transform
start_time = time.time()
best_fit = hough_transform_3d(points)
hough_time = time.time() - start_time

a_h, b_h, c_h, d_h = float(best_fit[0]), float(best_fit[1]), float(best_fit[2]), float(best_fit[3])


def evaluate_models(points, best_plane_hough):
    X = points[:, :2]
    true_z = points[:, 2]
    
    
    # Dự đoán z từ mô hình Hough
    a_h, b_h, c_h, d_h = best_plane_hough
    pred_z_hough = -(a_h * X[:, 0] + b_h * X[:, 1] + d_h) / c_h
    
    # Tính MSE cho từng mô hình

    mse_hough = mean_squared_error(true_z, pred_z_hough)
    
    return mse_hough

# Đánh giá các mô hình
mse_hough = evaluate_models(points, best_fit)

# In kết quả so sánh
print("\n=== So sánh các phương pháp ===")

print(f"Phương trình mặt phẳng từ Hough: z = {-a_h/c_h:.4f} * x + {-b_h/c_h:.4f} * y + {-d_h/c_h:.4f}")

print(f"Thời gian thực thi với Hough: {hough_time:.4f} giây")

print(f"MSE của Hough transform: {mse_hough:.4f}")