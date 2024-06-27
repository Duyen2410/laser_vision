import numpy as np

# Lực hấp dẫn giữa hai hạt
def gravitational_force(p1, p2, G=1.0):
    r = np.linalg.norm(p2 - p1)
    if r == 0:
        return np.zeros(3)
    force_magnitude = G / r**2
    direction = (p2 - p1) / r
    return force_magnitude * direction

# Cập nhật vị trí và vận tốc của các hạt
def update_positions_and_velocities(points, velocities, dt=0.01, G=1.0):
    n = points.shape[0]
    new_points = points.copy()
    new_velocities = velocities.copy()
    for i in range(n):
        force = np.zeros(3)
        for j in range(n):
            if i != j:
                force += gravitational_force(points[i], points[j], G)
        acceleration = force
        new_velocities[i] += acceleration * dt
        new_points[i] += new_velocities[i] * dt
    return new_points, new_velocities

# Tìm mặt phẳng từ các điểm ổn định
def fit_plane_to_points(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    U, S, Vt = np.linalg.svd(centered_points)
    normal = Vt[2]
    d = -np.dot(normal, centroid)
    return normal, d

# Hàm chính để tìm mặt phẳng tốt nhất
def find_best_plane(points, iterations=1000, dt=0.01, G=1.0):
    velocities = np.zeros(points.shape)
    for _ in range(iterations):
        points, velocities = update_positions_and_velocities(points, velocities, dt, G)
    normal, d = fit_plane_to_points(points)
    return normal, d

# Dữ liệu đầu vào
np.random.seed(0)
num_points = 1000
X = np.random.rand(num_points, 2) * 100
noise = np.random.randn(num_points) * 0.5
Z = 3 * X[:, 0] + 2 * X[:, 1] + 1 + noise
points = np.column_stack((X, Z))

# Tìm mặt phẳng tốt nhất
normal, d = find_best_plane(points)
print("Mặt phẳng tốt nhất có phương trình: {:.3f}x + {:.3f}y + {:.3f}z + {:.3f} = 0".format(normal[0], normal[1], normal[2], d))
