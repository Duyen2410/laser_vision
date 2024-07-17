import numpy as np
import time
from sklearn.metrics import mean_squared_error
import random
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs


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
    return fit_m.A1


    # Hàm để tính lỗi mô hình
def calculate_error(data, model):
    return np.mean([distance(point, model) for point in data])


def ransac_plane_fitting(points, n_per = 0.1):

    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    # Các thông số RANSAC
    n = int (n_per*len(points)) # Số mẫu tối thiểu để khớp mô hình
    t = 0.05*np.linalg.norm(points, axis=1).max()


    db = DBSCAN(eps=0.5, min_samples=5).fit(points)
    labels = db.labels_
    sorted_indices = np.argsort(labels)  # Sắp xếp theo nhãn
    points_sorted = points[sorted_indices]
   

    # Thuật toán RANSAC
    best_fit = None
    best_error = float('inf')
    max_inliers = 0
    iterations = 0
    no_improvement_counter = 0
    #points_sorted = np.asarray(sorted(points, key=lambda x: x[2]))  # Sort points by Z value
    i = 0

    while True:
        #sample = select_random(points, n)
        subset_indices = random.sample(range(min(i + n, len(points))), n)
        i += 10
        sample = points_sorted[subset_indices]
        model = fit(sample)
        distances = np.apply_along_axis(distance, 1, points, model)
        inliers = points[distances < t]

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            new_model = fit(inliers)
            new_error = calculate_error(inliers, new_model)
            if new_error < best_error:
                best_error = new_error
                best_fit = new_model
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
        else:
            no_improvement_counter += 1

        iterations += 1
        
        if iterations >= len(points) and no_improvement_counter >= 0.05*len(points):
            break

    return xs, ys, zs, best_fit, best_error, max_inliers


# Number of points
num_points = 1000

# Generate random 2D points
X = np.random.rand(num_points, 2) * 100

# Increase noise level
noise = np.random.randn(num_points) * 5  # Increased from 0.5 to 5

# Add nonlinear components
nonlinear_component1 = 10 * np.sin(0.1 * X[:, 0]) * np.cos(0.1 * X[:, 1])
nonlinear_component2 = 5 * np.log1p(X[:, 0])  # log1p(x) = log(1 + x)
nonlinear_component3 = 2 * (X[:, 0]**2) * np.sin(0.1 * X[:, 1])
nonlinear_component4 = 7 * np.exp(0.01 * X[:, 1])

# Calculate Z values with increased noise and nonlinear components
Z = (3 * X[:, 0] + 2 * X[:, 1] + 1 + noise +
     nonlinear_component1 + nonlinear_component2 +
     nonlinear_component3 + nonlinear_component4)

# Combine X and Z into a single dataset
points = np.column_stack((X, Z))


start_time = time.time()
xs, ys, zs, best_plane, best_error, max_inliers = ransac_plane_fitting(points)
custom_time = time.time() - start_time

a, b, c = best_plane

print(best_error)

def evaluate_models(points, best_plane_ransac):
    X = points[:, :2]
    true_z = points[:, 2]
    
    # Dự đoán z từ mô hình RANSAC
    a_r, b_r, c_r = best_plane_ransac
    pred_z_ransac = (a_r * X[:, 0] + b_r * X[:, 1] + c_r)
    mse_ransac = mean_squared_error(true_z, pred_z_ransac)
    return mse_ransac

mse_ransac = evaluate_models(points, best_plane)

print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {a:.4f} * x + {b:.4f} * y + {c:.4f}")

print(f"Thời gian thực thi với RANSAC tự triển khai: {custom_time:.4f} giây")

print(f"MSE của RANSAC tự triển khai: {mse_ransac:.4f}")

