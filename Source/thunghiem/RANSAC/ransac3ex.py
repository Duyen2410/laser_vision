import numpy as np
import time
from sklearn.metrics import mean_squared_error
import random

def ransac_plane_fitting(points):

    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
     # Các thông số RANSAC
    
    n = int (0.5*len(points)) # Số mẫu tối thiểu để khớp mô hình
    t = 0.2*np.linalg.norm(points, axis=1).max()
    k = int(0.1*len(points))  # Số lượng inliers tối thiểu
    m = int (0.1*len(points))

    def select_random(points, n):
        return points[random.sample(range(points.shape[0]), n)]

    # Hàm để tính khoảng cách từ điểm đến mô hình
    def distance(point, model):
        a, b, c = model
        x, y, z = point
        return np.abs(a * x + b * y - z + c) / np.sqrt(a**2 + b**2 + 1**2)

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
    return xs, ys, zs, best_fit, best_error


def evaluate_models(points, best_plane_ransac):
    X = points[:, :2]
    true_z = points[:, 2]
    
    # Dự đoán z từ mô hình RANSAC
    a_r, b_r, c_r = best_plane_ransac
    pred_z_ransac = a_r * X[:, 0] + b_r * X[:, 1] + c_r
    mse_ransac = mean_squared_error(true_z, pred_z_ransac)
    return mse_ransac



