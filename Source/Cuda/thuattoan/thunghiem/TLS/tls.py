import numpy as np
import time
from sklearn.metrics import mean_squared_error


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

np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))

start_time = time.time()
fit, errors, residual = fit_plane_tls(points)
tls_time = time.time() - start_time

a_tls, b_tls, c_tls = float(fit[0]), float(fit[1]), float(fit[2])

def evaluate_models(points, best_plane_tls):
    X = points[:, :2]
    true_z = points[:, 2]
    
    
    # Dự đoán z từ mô hình TLS
    a_tls, b_tls, c_tls = best_plane_tls
    pred_z_tls = a_tls * X[:, 0] + b_tls * X[:, 1] + c_tls
    
    
    
    # Tính MSE cho từng mô hình
    mse_tls = mean_squared_error(true_z, pred_z_tls)
    
    return mse_tls

mse_tls = evaluate_models(points, (a_tls, b_tls, c_tls))
print(f"Phương trình mặt phẳng từ TLS: z = {a_tls:.4f} * x + {b_tls:.4f} * y + {c_tls:.4f}")
print(f"Thời gian thực thi với TLS: {tls_time:.4f} giây")
print(f"MSE của TLS: {mse_tls:.4f}")