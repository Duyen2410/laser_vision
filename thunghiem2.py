import numpy as np
import time
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

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

def ransac_plane_fitting(points, iterations=1000, distance_threshold=1.0):
    def fit_plane(points):
        p1, p2, p3 = points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        a, b, c = normal
        d = -np.dot(normal, p1)
        return a, b, c, d

    def calculate_distance(points, a, b, c, d):
        return np.abs(a * points [0] + b * points [1] + c * points [2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

    best_inliers = []
    best_plane = None

    for _ in range(iterations):
        sample_points = points[np.random.choice(points.shape[0], 3, replace=False)]
        a, b, c, d = fit_plane(sample_points)

        inliers = []
        for point in points:
            distance = calculate_distance(point, a, b, c, d)
            if distance < distance_threshold:
                inliers.append(point)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)

    return best_plane, best_inliers



import numpy as np
from itertools import product

def hough_transform_3d(points, theta_res=0.1, phi_res=0.1, d_res=0.1):
    points = np.asarray(points)
    
    # Define the ranges for theta, phi, and d
    theta_max = np.pi
    phi_max = 2 * np.pi
    d_max = np.linalg.norm(points, axis=1).max()
    
    # Create the accumulator array
    theta_bins = int(theta_max / theta_res)
    phi_bins = int(phi_max / phi_res)
    d_bins = int((2 * d_max) / d_res)
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
num_points = 100
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))


# So sánh các phương pháp


# Phương pháp tự triển khai RANSAC
start_time = time.time()
best_plane, best_inliers = ransac_plane_fitting(points)
custom_time = time.time() - start_time

a, b, c, d = best_plane

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



# In kết quả so sánh
print("\n=== So sánh các phương pháp ===")

print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {-a/c:.4f} * x + {-b/c:.4f} * y + {-d/c:.4f}")
print(f"Phương trình mặt phẳng từ TLS: z = {a_tls:.4f} * x + {b_tls:.4f} * y + {c_tls:.4f}")
print(f"Phương trình mặt phẳng từ Hough: z = {-a_h/c_h:.4f} * x + {-b_h/c_h:.4f} * y + {-d_h/c_h:.4f}")

print(f"Thời gian thực thi với RANSAC tự triển khai: {custom_time:.4f} giây")
print(f"Thời gian thực thi với TLS: {tls_time:.4f} giây")
print(f"Thời gian thực thi với Hough: {hough_time:.4f} giây")

