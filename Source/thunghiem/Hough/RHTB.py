import numpy as np
import random
from scipy.ndimage import maximum_filter
from sklearn.preprocessing import normalize
import time
from sklearn.metrics import mean_squared_error


import numpy as np

def detect_best_plane(D, k_max=1000, n = 100, theta_res=0.05, phi_res=0.05, rho_res=0.05):
    points = np.asarray(D)
    theta_max = np.pi
    phi_max = 2 * np.pi
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
        a, b, c, d = model
        rho = d / np.linalg.norm(model[:3])
        theta = np.arccos(c / np.linalg.norm(model[:3]))  # Calculate theta using arccos
        phi = np.arctan2(b, a)  # Calculate phi using arctan2
        return theta, phi, rho

    for k in range(k_max):
        sample = select_random(points, n)
        model = fit(sample)
        theta, phi, rho = model
        theta_idx = int(theta / theta_res)
        phi_idx = int(phi / phi_res)
        rho_idx = int((rho + rho_max) / rho_res)
        A[theta_idx, phi_idx, rho_idx] += 1

      
    idx = np.unravel_index(np.argmax(A), A.shape)

    theta_idx, phi_idx, rho_idx = idx
    best_theta = theta_idx * theta_res
    best_phi = phi_idx * phi_res
    best_rho = rho_idx * rho_res - rho_max
    a = np.sin(best_theta) * np.cos(best_phi)
    b = np.sin(best_theta) * np.sin(best_phi)
    c = np.cos(best_theta)
    best_fit_planes = (a, b, c, best_rho)

    return best_fit_planes


np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1000 + noise
points = np.column_stack((X, Z))

# Phương pháp Hough transform
start_time = time.time()
best_fit = detect_best_plane(points, k_max= 400, n = 800, theta_res=0.05, phi_res=0.05, rho_res=0.05)
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





