import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from itertools import product
import random
from scipy.ndimage import gaussian_filter

def ransac_plane_fitting(points):

    def select_random(points, n):
        return points[random.sample(range(points.shape[0]), n)]

    # Hàm để tính khoảng cách từ điểm đến mô hình
    def distance(point, model):
        a, b, c = model
        x, y, z = point
        return np.abs(a * x + b * y + z + c) / np.sqrt(a**2 + b**2 + 1**2)

    # Hàm để khớp mô hình với các điểm dữ liệu
    def fit(points):
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
        fit_m = (A.T * A).I * A.T * b
        errors = b - A * fit_m
        return fit_m.A1

    # Hàm để tính lỗi mô hình
    def calculate_error(data, model):
       return np.mean([distance(point, model) for point in data])


    # Các thông số RANSAC
    n = 200  # Số mẫu tối thiểu để khớp mô hình
    t = 5 # Ngưỡng khoảng cách
    k = 250  # Số lượng inliers tối thiểu
    m = 2000 # Số lần lặp lại

    # Thuật toán RANSAC
    best_fit = None
    best_error = float('inf')
    max_in = 0
    for i in range(m):
        temp_inliers = select_random(points, n)
        temp_inliers_add = []
        temp_model = fit(temp_inliers)

        for point in points:
            if point.tolist() not in temp_inliers.tolist():
                if distance(point, temp_model) < t:
                    print(distance(point, temp_model))
                    temp_inliers_add.append(point)
        temp_inliers_add = np.array(temp_inliers_add)
        if len(temp_inliers_add) > max_in:
            max_in = len(temp_inliers_add)
            all_inliers = np.vstack((temp_inliers, temp_inliers_add))
            new_model = fit(all_inliers)
            new_error = calculate_error(all_inliers, new_model)
            if new_error < best_error:
                best_error = new_error
                best_fit = new_model
        
    print(best_fit)
    return best_fit, best_error

np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))


start_time = time.time()
best_plane, best_error = ransac_plane_fitting(points)
ransac_time = time.time() - start_time

a, b, c = best_plane

print(best_error)

def evaluate_models(points, best_plane_ransac):
    X = points[:, :2]
    true_z = points[:, 2]
    
    # Dự đoán z từ mô hình RANSAC
    a_r, b_r, c_r = best_plane_ransac
    pred_z_ransac = a_r * X[:, 0] + b_r * X[:, 1] + c_r
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

mse_ransac, r_adj_sq, mae_ransac = evaluate_models(points, best_plane)
print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {a:.4f} * x + {b:.4f} * y + {c:.4f}")
print(f"Thời gian Calib_laser với RANSAC: {ransac_time:.4f} giây")
print(f"MSE của RANSAC tự triển khai: {mse_ransac:.4f}")
print(f"R^2 : {r_adj_sq:.4f}")
print(f"MAE: {mae_ransac:.4f}")

