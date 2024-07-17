import numpy as np
import time
from sklearn.metrics import mean_squared_error
from numba import cuda, float32
import math

@cuda.jit
def hough_transform_3d_kernel(points, accumulator, theta_res, phi_res, rho_res, rho_max):
    # Tính toán chỉ số toàn cục của luồng hiện tại
    theta_idx, phi_idx = cuda.grid(2)

    theta_max = np.pi
    phi_max = 2 * np.pi
    rho_bins = accumulator.shape[2]
    
    if theta_idx < accumulator.shape[0] and phi_idx < accumulator.shape[1]:
        theta = theta_idx * theta_res
        phi = phi_idx * phi_res

        for i in range(points.shape[0]):
            x, y, z = points[i]
            a = math.sin(theta) * math.cos(phi)
            b = math.sin(theta) * math.sin(phi)
            c = math.cos(theta)
            rho = a * x + b * y + c * z
            rho_idx = int((rho + rho_max) / rho_res)
            if 0 <= rho_idx < rho_bins:
                cuda.atomic.add(accumulator, (theta_idx, phi_idx, rho_idx), 1)

def hough_transform_3d(points, theta_res=0.1, phi_res=0.1, rho_res=0.1):
    points = np.asarray(points, dtype=np.float32)

    # Define the ranges for theta, phi, and d
    theta_max = np.pi
    phi_max = 2 * np.pi
    rho_max = np.linalg.norm(points, axis=1).max()
    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    rho_bins = int((2 * rho_max) / rho_res)
    
    accumulator = np.zeros((theta_bins, phi_bins, rho_bins), dtype=np.int32)
    d_accumulator = cuda.to_device(accumulator)
    d_points = cuda.to_device(points)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(theta_bins / threadsperblock[0])
    blockspergrid_y = math.ceil(phi_bins / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    hough_transform_3d_kernel[blockspergrid, threadsperblock](d_points, d_accumulator, theta_res, phi_res, rho_res, rho_max)

    accumulator = d_accumulator.copy_to_host()

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