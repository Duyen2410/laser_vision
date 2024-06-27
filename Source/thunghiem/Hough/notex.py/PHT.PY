import numpy as np
import time
from sklearn.metrics import mean_squared_error
import random
def p_hough_transform_3d(points, theta_res=0.05, phi_res=0.05, rho_res=0.05, points_num = 600, threshold = 10):
    points = np.asarray(points)
    p_m = random.sample(list(points), points_num)
    # Define the ranges for theta, phi, and d
    theta_max = np.pi
    phi_max = 2 * np.pi
    rho_max = np.linalg.norm(p_m, axis=1).max()
    
    # Create the accumulator array
    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    rho_bins = int((2*rho_max) / rho_res)
    accumulator = np.zeros((theta_bins, phi_bins, rho_bins))


    for x, y, z in p_m:
        for theta_idx in range(theta_bins):
            theta = theta_idx * theta_res
            for phi_idx in range(phi_bins):
                phi = phi_idx * phi_res
                a = np.sin(theta) * np.cos(phi)
                b = np.sin(theta) * np.sin(phi)
                c = np.cos(theta)
                rho = a * x + b * y + c * z
                rho_idx = int((rho + rho_max)/theta_res)
                if 0 <= rho_idx < rho_bins:
                    accumulator[theta_idx, phi_idx, rho_idx] += 1

    #accumulator[accumulator < threshold] = 0

    idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    theta_idx_f, phi_idx_f, rho_idx_f = idx
    best_theta = theta_idx_f * theta_res
    best_phi = phi_idx_f * phi_res
    best_rho = rho_idx_f * rho_res - rho_max

    a = np.sin(best_theta) * np.cos(best_phi)
    b = np.sin(best_theta) * np.sin(best_phi)
    c = np.cos(best_theta)

    best_fit_plane = (a, b, c, -best_rho)
    return best_fit_plane

np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))

# Phương pháp Hough transform
start_time = time.time()
best_fit = p_hough_transform_3d(points)
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