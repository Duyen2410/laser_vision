import numpy as np
import time
from numba import cuda, float32, int32
from sklearn.metrics import mean_squared_error

# Hàm thiết bị để tính khoảng cách từ một điểm đến một mặt phẳng
@cuda.jit(device=True)
def calculate_distance(x, y, z, A, B, C, D):
    return abs(A * x + B * y + C * z + D) / ((A**2 + B**2 + C**2)**0.5)

# Hàm thiết bị để xáo trộn chỉ số sử dụng thuật toán Fisher-Yates
@cuda.jit(device=True)
def shuffle_indices(indices, n, seed):
    for i in range(n-1, 0, -1):
        j = (seed + i) % n
        indices[i], indices[j] = indices[j], indices[i]

# Kernel con để tính toán mặt phẳng và đếm inliers
@cuda.jit(device=True)
def compute_plane(points, indices, threshold, n, best_plane, best_inliers, best_residual, seed):
    p1, p2, p3 = points[indices[0]], points[indices[1]], points[indices[2]]

    # Tính toán mặt phẳng
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    
    # Vectơ của các cạnh của tam giác
    v1 = (x2 - x1, y2 - y1, z2 - z1)
    v2 = (x3 - x1, y3 - y1, z3 - z1)
    
    # Tính toán vectơ pháp tuyến của mặt phẳng
    normal = (
        v1[1] * v2[2] - v1[2] * v2[1],  # A
        v1[2] * v2[0] - v1[0] * v2[2],  # B
        v1[0] * v2[1] - v1[1] * v2[0]   # C
    )
    
    A, B, C = normal
    D = -(A * x1 + B * y1 + C * z1)
    
    # Đo lường và xác định các điểm inliers
    inliers = 0
    residual_sum = 0
    for i in range(n):
        x, y, z = points[i]
        distance = calculate_distance(x, y, z, A, B, C, D)
        if distance < threshold:
            inliers += 1
            residual_sum += distance ** 2
    
    # Kiểm tra và cập nhật mặt phẳng tốt nhất
    if inliers > best_inliers[0]:
        best_inliers[0] = inliers
        best_plane[0] = A
        best_plane[1] = B
        best_plane[2] = C
        best_plane[3] = D
        best_residual[0] = residual_sum

# Kernel chính sử dụng phân luồng nhiều lớp
@cuda.jit
def ransac_kernel(points, best_plane, best_inliers, best_residual, max_iterations, threshold, n, seed):
    # Lấy chỉ số của luồng
    idx = cuda.grid(1)
    if idx >= max_iterations:
        return
    
    # Khởi tạo bộ nhớ cục bộ cho các chỉ số điểm
    indices = cuda.local.array(1000, dtype=int32)
    for i in range(n):
        indices[i] = i
    
    # Xáo trộn chỉ số điểm sử dụng thuật toán Fisher-Yates
    shuffle_indices(indices, n, seed + idx)
    
    # Gọi kernel con để tính toán mặt phẳng và đếm inliers
    compute_plane(points, indices, threshold, n, best_plane, best_inliers, best_residual, seed + idx)

# Hàm chính để gọi kernel CUDA
def ransac(points, max_iterations=1000, threshold=0.05):
    n = points.shape[0]
    # Chuyển đổi dữ liệu thành mảng numpy và chuẩn bị cho CUDA
    points_device = cuda.to_device(points.astype(np.float32))
    best_plane = cuda.device_array(4, dtype=np.float32)
    best_inliers = cuda.device_array(1, dtype=np.int32)
    best_residual = cuda.device_array(1, dtype=np.float32)
    
    # Khởi tạo giá trị ban đầu
    best_inliers[0] = 0
    best_residual[0] = float('inf')
    
    # Cấu hình cho kernel
    threads_per_block = 32
    blocks_per_grid = (max_iterations + (threads_per_block - 1)) // threads_per_block
    
    start_time = time.time()
    
    # Tạo một seed ngẫu nhiên
    seed = np.random.randint(0, 1000000)
    
    # Gọi kernel CUDA
    ransac_kernel[blocks_per_grid, threads_per_block](points_device, best_plane, best_inliers, best_residual, max_iterations, threshold, n, seed)
    
    cuda.synchronize()
    
    end_time = time.time()
    
    # Lấy kết quả từ device
    best_plane = best_plane.copy_to_host()
    best_inliers = best_inliers.copy_to_host()
    best_residual = best_residual.copy_to_host()
    
    A, B, C, D = best_plane
    return (A, B, C, D), best_inliers[0], best_residual[0], end_time - start_time

# Tạo dữ liệu điểm và chạy RANSAC
np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 100 + noise
points = np.column_stack((X, Z))

# Chạy RANSAC để tìm mặt phẳng tốt nhất
best_plane, best_inliers, best_residual, ransac_time = ransac(points, max_iterations=1000, threshold=0.5)

a_ransac, b_ransac, c_ransac, d_ransac = best_plane

# Tính toán MSE cho mặt phẳng RANSAC
X = points[:, :2]
true_z = points[:, 2]
pred_z_ransac = -(a_ransac * X[:, 0] + b_ransac * X[:, 1] + d_ransac) / c_ransac
mse_ransac = mean_squared_error(true_z, pred_z_ransac)

print(f"Phương trình mặt phẳng từ RANSAC: z = {-a_ransac/c_ransac:.4f} * x + {-b_ransac/c_ransac:.4f} * y + {-d_ransac/c_ransac:.4f}")
print(f"Thời gian thực thi với RANSAC: {ransac_time:.4f} giây")
print(f"MSE của RANSAC: {mse_ransac:.4f}")
