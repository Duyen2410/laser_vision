import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error
import numpy as np

def detect_best_plane(D, k_max=5000, n = 100, theta_res=0.05, phi_res=0.05, rho_res=0.05):

    points = np.asarray(D)
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    theta_max = 2*np.pi
    phi_max = np.pi
    rho_max = np.linalg.norm(points, axis=1).max()

    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    rho_bins = int((2 * rho_max) / rho_res)
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
        norm = np.linalg.norm(model[:3])
        a, b, c, d = model / norm
        rho = -d
        phi = np.arccos(c)  # Calculate theta using arccos
        theta = np.arctan2(b, a)  # Calculate phi using arctan2
        return theta, phi, rho, model

    for k in range(k_max):
        sample = select_random(points, n)
        theta, phi, rho, model = fit(sample)
        residuals = np.abs(xs * np.sin(phi) * np.cos(theta) + ys * np.sin(phi) * np.sin(theta) + zs * np.cos(phi) - rho)
        inliers = residuals < 0.01  # Threshold for inliers
        if np.sum(inliers) < n * 0.05:  # Skip if not enough inliers
            continue
        theta_idx = int(theta / theta_res)
        phi_idx = int(phi / phi_res)
        rho_idx = int((rho + rho_max) / rho_res)
        if 0 <= rho_idx < rho_bins and 0 <= theta_idx < theta_bins and 0 <= phi_idx < phi_bins:
            weight = np.exp(-np.mean(residuals[inliers]**2))
            A[theta_idx, phi_idx, rho_idx] += weight

      
    idx = np.unravel_index(np.argmax(A), A.shape)
    theta_idx, phi_idx, rho_idx = idx
    # Calculate averages
    best_theta = theta_idx * theta_res
    best_phi = phi_idx * phi_res
    best_rho = rho_idx * rho_res - rho_max
    a = np.sin(best_phi) * np.cos(best_theta)
    b = np.sin(best_phi) * np.sin(best_theta)
    c = np.cos(best_phi)
    best_fit_planes = (a, b, c, -best_rho)

    return xs, ys, zs, best_fit_planes


np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 6 * X[:, 1] + 100 + noise
points = np.column_stack((X, Z))

# Phương pháp Hough transform
start_time = time.time()
_, _, _, best_fit = detect_best_plane(points, k_max= 5000, n = 100, theta_res=0.1, phi_res=0.1, rho_res=0.1)
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





