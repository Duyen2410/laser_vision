import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Khai báo Bảng DH cho Robot 6DOF
class Robot:
    def __init__(self, dh_params):
        self.dh_params = dh_params

    def dh_matrix(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            T_i = self.dh_matrix(joint_angles[i] + theta, d, a, alpha)
            T = np.dot(T, T_i)
        return T

# 2. Ma trận biến đổi từ base đến đầu hàn
# Giả sử các tham số DH là [theta, d, a, alpha] cho từng khớp
dh_params = [
    [0, 0, 0.1, np.pi/2],
    [0, 0, 0.5, 0],
    [0, 0, 0.3, np.pi/2],
    [0, 0.5, 0, -np.pi/2],
    [0, 0, 0, np.pi/2],
    [0, 0.1, 0, 0]
]

robot = Robot(dh_params)

# Giả sử tại vị trí ban đầu các khớp đều bằng 0
joint_angles = [0, 0, 0, 0, 0, 0]
T_base_to_welding = robot.forward_kinematics(joint_angles)

# 3. Ma trận Jacobian và Jacobian đảo
def jacobian(joint_angles, dh_params):
    # Tính toán ma trận Jacobian
    pass

def inverse_jacobian(joint_angles, dh_params):
    J = jacobian(joint_angles, dh_params)
    return np.linalg.pinv(J)

J_inv = inverse_jacobian(joint_angles, dh_params)

# 4. Trích xuất Thông tin từ Camera
def extract_welding_point(image):
    # Code để trích xuất tọa độ điểm hàn từ hình ảnh
    welding_point = [x, y, z] # Đơn vị: pixel hoặc tọa độ hình ảnh
    return welding_point

def compute_welding_orientation(welding_point):
    # Tính toán hướng của đầu mỏ hàn
    normal_vector = [0, 0, 1] # Z của camera là vécto pháp tuyến của mặt phẳng hàn
    tangent_vector = [0, 1, 0] # Y là đạo hàm đường hàn
    cross_vector = np.cross(normal_vector, tangent_vector) # Z là tích có hướng
    orientation_matrix = np.array([tangent_vector, cross_vector, normal_vector]).T
    return orientation_matrix

# 5. Chuyển Tọa độ sang Không gian Thực
def pixel_to_real(pixel_coords, intrinsic_matrix, extrinsic_matrix):
    # Chuyển tọa độ pixel sang không gian thực
    pass

# Ma trận thông số nội (intrinsic) và thông số ngoại (extrinsic) đã biết
intrinsic_matrix = np.array([[...], [...], [...]])
extrinsic_matrix = np.array([[...], [...], [...]])

# 6. Lưu Ma trận vào Các Danh sách
T_welding_to_base_list = []
extrinsic_matrix_list = []

def update_matrices(joint_angles, welding_point):
    T_welding = robot.forward_kinematics(joint_angles)
    T_welding_to_base_list.append(T_welding)

    real_coords = pixel_to_real(welding_point, intrinsic_matrix, extrinsic_matrix)
    extrinsic_matrix_list.append(extrinsic_matrix)

# 7. Di chuyển Robot
def move_robot(joint_angles):
    # Hàm để điều khiển robot di chuyển
    pass

# Giả sử có một danh sách các điểm hàn trên đường hàn
welding_points = [ ... ] 

for point in welding_points:
    welding_point = extract_welding_point(point)
    orientation_matrix = compute_welding_orientation(welding_point)
    
    # Cập nhật ma trận
    update_matrices(joint_angles, welding_point)
    
    # Điều khiển robot đến vị trí tiếp theo
    joint_angles = J_inv @ np.array(welding_point)
    move_robot(joint_angles)
