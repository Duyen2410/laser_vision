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
        a, b, c, d = model
        x, y, z = point
        return np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

    # Hàm để khớp mô hình với các điểm dữ liệu
    def fit(points):
        xs = points[:,0]
        ys = points[:,1]
        zs = points[:,2]
        tmp_A = np.c_[xs, ys, zs, np.ones(xs.shape[0])]
        U, S, Vt = np.linalg.svd(tmp_A)
        model = Vt[-1, :] # Giải pháp là hàng cuối cùng của Vt
        return model

    # Hàm để tính lỗi mô hình
    def calculate_error(data, model):
       return np.mean([distance(point, model) for point in data])


    # Các thông số RANSAC
    n = 255  # Số mẫu tối thiểu để khớp mô hình
    t = 0.05 # Ngưỡng khoảng cách
    k = 250  # Số lượng inliers tối thiểu
    m = 1000 # Số lần lặp lại

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
custom_time = time.time() - start_time

a, b, c, d = best_plane

print(best_error)

def evaluate_models(points, best_plane_ransac):
    X = points[:, :2]
    true_z = points[:, 2]
    
    # Dự đoán z từ mô hình RANSAC
    a_r, b_r, c_r, d_r = best_plane_ransac
    pred_z_ransac = -(a_r * X[:, 0] + b_r * X[:, 1] + d_r)/c_r
    mse_ransac = mean_squared_error(true_z, pred_z_ransac)
    return mse_ransac

mse_ransac = evaluate_models(points, best_plane)

print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {-a/c:.4f} * x + {-b/c:.4f} * y + {-d/c:.4f}")

print(f"Thời gian thực thi với RANSAC tự triển khai: {custom_time:.4f} giây")

print(f"MSE của RANSAC tự triển khai: {mse_ransac:.4f}")

