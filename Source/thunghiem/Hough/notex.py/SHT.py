import numpy as np
import time
from sklearn.metrics import mean_squared_error

def hough_transform_3d(points, theta_res=0.1, phi_res=0.1, rho_res=0.1):
    points = np.asarray(points)
    

    # Define the ranges for theta, phi, and d
    theta_max = np.pi
    phi_max = 2 * np.pi
    rho_max = np.linalg.norm(points, axis=1).max()
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

mse_hough, r_adj_sq, mae_hough = evaluate_models(points, best_fit)
#print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {a:.4f} * x + {b:.4f} * y + {c:.4f}")
print(f"Thời gian Calib_laser với RANSAC: {hough_time:.4f} giây")
print(f"MSE của RANSAC tự triển khai: {mse_hough:.4f}")
print(f"R^2 : {r_adj_sq:.4f}")
print(f"MAE: {mae_hough:.4f}")