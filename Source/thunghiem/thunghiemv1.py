import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from itertools import product
import random

def fit_plane_tls(points):
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    return fit, errors, residual

def ransac_plane_fitting(points):

    # Hàm để chọn ngẫu nhiên n điểm từ dữ liệu
    def select_random(points, n):
        return points[random.sample(range(points.shape[0]), n)]

    # Hàm để tính khoảng cách từ điểm đến mô hình
    def distance(point, model):
        a, b, c, d = model
        x, y, z = point
        return np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

    # Hàm để khớp mô hình với các điểm dữ liệu
    def fit(data):
        X = data[:, 0]
        Y = data[:, 1]
        Z = data[:, 2]
        A = np.c_[X, Y, Z, np.ones(X.shape[0])]
        U, S, Vt = np.linalg.svd(A)
        model = Vt[-1]
        return model / model[-1]  # Chuẩn hóa để d = 1

    # Hàm để tính lỗi mô hình
    def calculate_error(data, model):
        return np.mean([distance(point, model) for point in data])


    # Các thông số RANSAC
    n = 2  # Số mẫu tối thiểu để khớp mô hình
    t = 10  # Ngưỡng sai số
    k = 100  # Số lượng inliers tối thiểu
    m = 10000  # Số lần lặp lại

    # Thuật toán RANSAC
    best_fit = None
    best_error = float('inf')

    for i in range(m):
        temp_inliers = select_random(points, n)
        temp_inliers_add = []

        temp_model = fit(temp_inliers)

        for point in points:
            if point.tolist() not in temp_inliers.tolist():
                if distance(point, temp_model) < t:
                    temp_inliers_add.append(point)

        temp_inliers_add = np.array(temp_inliers_add)

        if len(temp_inliers_add) > k:
            all_inliers = np.vstack((temp_inliers, temp_inliers_add))
            new_model = fit(all_inliers)
            new_error = calculate_error(all_inliers, new_model)

            if new_error < best_error:
                best_error = new_error
                best_fit = new_model

    return best_fit, best_error



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


# Tạo ra một tập hợp các điểm 3D với một mặt phẳng đã biết và nhiễu ngẫu nhiên
np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
linear_noise = np.random.randn(num_points) * 0.5
nonlinear_noise1 = np.sin(X[:, 0] / 10) * np.cos(X[:, 1] / 10) * 20
nonlinear_noise2 = np.sin(X[:, 0] / 5) * np.cos(X[:, 1] / 5) * 10
random_noise1 = np.random.randn(num_points) * 0.5
random_noise2 = np.random.rand(num_points) * 1.0
random_noise3 = np.random.laplace(0, 0.5, num_points) * 1.5
Z = (
    3 * X[:, 0]
    + 2 * X[:, 1]
    + 500
    + linear_noise
    + nonlinear_noise1
    + nonlinear_noise2
    + random_noise1
    + random_noise2
    + random_noise3
)

# Kết hợp các điểm dữ liệu lại
points = np.column_stack((X, Z))


# So sánh các phương pháp


# Phương pháp tự triển khai RANSAC
start_time = time.time()
best_plane, best_inliers = ransac_plane_fitting(points)
custom_time = time.time() - start_time

a, b, c = best_plane

# Phương pháp TLS
start_time = time.time()
fit, errors, residual = fit_plane_tls(points)
tls_time = time.time() - start_time

a_tls, b_tls, c_tls = float(fit[0]), float(fit[1]), float(fit[2])


# Phương pháp Hough transform
start_time = time.time()
best_fit = hough_transform_3d(points)
hough_time = time.time() - start_time

a_h, b_h, c_h, d_h = float(best_fit[0]), float(best_fit[1]), float(best_fit[2]), float(best_fit[3])

from sklearn.metrics import mean_squared_error

def evaluate_models(points, best_plane_ransac, best_plane_tls, best_plane_hough):
    X = points[:, :2]
    true_z = points[:, 2]
    
    # Dự đoán z từ mô hình RANSAC
    a_r, b_r, c_r, d_r = best_plane_ransac
    pred_z_ransac = -(a_r * X[:, 0] + b_r * X[:, 1] + d_r) / c_r
    
    # Dự đoán z từ mô hình TLS
    a_tls, b_tls, c_tls = best_plane_tls
    pred_z_tls = a_tls * X[:, 0] + b_tls * X[:, 1] + c_tls
    
    # Dự đoán z từ mô hình Hough
    a_h, b_h, c_h, d_h = best_plane_hough
    pred_z_hough = -(a_h * X[:, 0] + b_h * X[:, 1] + d_h) / c_h
    
    # Tính MSE cho từng mô hình
    mse_ransac = mean_squared_error(true_z, pred_z_ransac)
    mse_tls = mean_squared_error(true_z, pred_z_tls)
    mse_hough = mean_squared_error(true_z, pred_z_hough)
    
    return mse_ransac, mse_tls, mse_hough

# Đánh giá các mô hình
mse_ransac, mse_tls, mse_hough = evaluate_models(points, best_plane, (a_tls, b_tls, c_tls), best_fit)

# In kết quả so sánh
print("\n=== So sánh các phương pháp ===")

print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {-a/c:.4f} * x + {-b/c:.4f} * y + {-d/c:.4f}")
print(f"Phương trình mặt phẳng từ TLS: z = {a_tls:.4f} * x + {b_tls:.4f} * y + {c_tls:.4f}")
print(f"Phương trình mặt phẳng từ Hough: z = {-a_h/c_h:.4f} * x + {-b_h/c_h:.4f} * y + {-d_h/c_h:.4f}")

print(f"Thời gian thực thi với RANSAC tự triển khai: {custom_time:.4f} giây")
print(f"Thời gian thực thi với TLS: {tls_time:.4f} giây")
print(f"Thời gian thực thi với Hough: {hough_time:.4f} giây")

print("\n=== Đánh giá độ chính xác các phương pháp ===")
print(f"MSE của RANSAC tự triển khai: {mse_ransac:.4f}")
print(f"MSE của TLS: {mse_tls:.4f}")
print(f"MSE của Hough transform: {mse_hough:.4f}")