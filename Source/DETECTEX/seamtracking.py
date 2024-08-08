import numpy as np
import cv2
import time

class OptimizedRobot:
    def __init__(self, d, a, alpha, T_welding_to_camera, intrinsic_matrix):
        self.d = np.array(d)
        self.a = np.array(a)
        self.alpha = np.array(alpha)
        self.T_welding_to_camera = T_welding_to_camera
        self.intrinsic_matrix = intrinsic_matrix

    def dh_matrix(self, theta, d, a, alpha):
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

        return np.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        T_list = [T]
        for theta, d, a, alpha in zip(joint_angles, self.d, self.a, self.alpha):
            T_i = self.dh_matrix(theta, d, a, alpha)
            T = T @ T_i
            T_list.append(T)
        return T, T_list

    def jacobian(self, joint_angles):
        _, T_list = self.forward_kinematics(joint_angles)
        J = np.zeros((6, len(joint_angles)))

        end_effector_pos = T_list[-1][:3, 3]

        for i in range(len(joint_angles)):
            z_i = T_list[i][:3, 2]
            o_i = T_list[i][:3, 3]

            J[:3, i] = np.cross(z_i, end_effector_pos - o_i)
            J[3:, i] = z_i

        return J

    def inverse_jacobian(self, joint_angles):
        J = self.jacobian(joint_angles)
        try:
            J_inv = np.linalg.pinv(J, rcond=1e-4)
        except np.linalg.LinAlgError:
            J_inv = np.linalg.pinv(J)
        return J_inv

    def compute_external_parameters_matrix(self, T_base_to_welding):
        return T_base_to_welding @ self.T_welding_to_camera

    def detect_joint_edge(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        edge_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                edge_points.append(((x1, y1), (x2, y2)))
        return edge_points

    def find_knee_points(self, edge_points):
        knee_points = []

        for i in range(1, len(edge_points) - 1):
            (x1, y1), (x2, y2) = edge_points[i]
            (x3, y3), (x4, y4) = edge_points[i + 1]

            slope1 = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            slope2 = (y4 - y3) / (x4 - x3) if x4 != x3 else np.inf

            if np.abs(slope1 - slope2) > 0.1:
                knee_points.append((x2, y2))

        return knee_points

    def pixel_to_real_world(self, x, y):
        point_homogeneous = np.array([x, y, 1])
        point_3d_camera = np.linalg.inv(self.intrinsic_matrix) @ point_homogeneous
        point_3d_world = np.linalg.inv(self.T_welding_to_camera) @ np.append(point_3d_camera, 1)

        return point_3d_world[:3]

    def find_p_q_points(self, knee_points, offset=5):
        p_q_points = []

        for (x, y) in knee_points:
            direction_vector = np.array([y, -x])
            normalized_vector = direction_vector / np.linalg.norm(direction_vector)

            point_real_world = self.pixel_to_real_world(x, y)

            p_real_world = point_real_world + offset * normalized_vector

            p_q_points.append(p_real_world)

        return p_q_points

    def find_welding_point(self, u_real_world, v_real_world):
        welding_point = (u_real_world + v_real_world) / 2
        return welding_point

def find_normal_vector(p, q, r):
    pq = q - p
    pr = r - p

    normal_vector = np.cross(pq, pr)

    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    normal_vector = normal_vector / np.linalg.nor/m(normal_vector)

    return normal_vector

def compute_movement_vector(current_point, next_point):
    movement_vector = next_point - current_point
    movement_vector = movement_vector / np.linalg.norm(movement_vector)
    return movement_vector

def compute_cross_product_vector(movement_vector, normal_vector):
    cross_product_vector = np.cross(movement_vector, normal_vector)
    cross_product_vector = cross_product_vector / np.linalg.norm(cross_product_vector)
    return cross_product_vector

def welding_to_base_matrix(normal_vector, movement_vector, cross_product_vector, welding_point):
    rotation_matrix = np.array([
        cross_product_vector,
        movement_vector,
        normal_vector
    ]).T

    T_base_welding = np.eye(4)
    T_base_welding[:3, :3] = rotation_matrix
    T_base_welding[:3, 3] = welding_point

    return T_base_welding

def optimized_seam_tracking(T_star, T_6i, q_i, d, a, alpha, T_welding_to_camera, intrinsic_matrix, max_iterations=100, tolerance=1e-4):
    robot = OptimizedRobot(d, a, alpha, T_welding_to_camera, intrinsic_matrix)
    previous_dq_i_plus_1 = np.zeros(len(q_i))  # Khởi tạo biến lưu trữ thay đổi khớp trước đó để kiểm tra hội tụ
    for i in range(max_iterations):
        dT_6 = T_star - T_6i
        J_inv = robot.inverse_jacobian(q_i)

        dq_i_plus_1 = J_inv @ dT_6.flatten()[:6]

        # Giới hạn sự thay đổi của các khớp
        dq_i_plus_1 = np.clip(dq_i_plus_1, -0.1, 0.1)

        # Kiểm tra hội tụ dựa trên thay đổi khớp và sai số trong không gian Cartesian
        if np.linalg.norm(dq_i_plus_1 - previous_dq_i_plus_1) < tolerance and np.linalg.norm(dT_6) < tolerance:
            print(f"Thuật toán hội tụ sau {i+1} vòng lặp.")
            break

        q_i_plus_1 = q_i + dq_i_plus_1
        T_6i_new, _ = robot.forward_kinematics(q_i_plus_1)

        previous_dq_i_plus_1 = dq_i_plus_1
        q_i = q_i_plus_1
        T_6i = T_6i_new

    return q_i, T_6i

def real_time_welding(robot, T_star, initial_joint_angles, camera, d, a, alpha, T_welding_to_camera, intrinsic_matrix):
    q_i = initial_joint_angles
    T_6i, _ = robot.forward_kinematics(q_i)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        edge_points = robot.detect_joint_edge(frame)
        if edge_points:
            knee_points = robot.find_knee_points(edge_points)
            if knee_points:
                p_q_points = robot.find_p_q_points(knee_points)
                if len(p_q_points) >= 2:
                    u_real_world, v_real_world = p_q_points[0], p_q_points[1]
                    welding_point = robot.find_welding_point(u_real_world, v_real_world)

                    # Calculate normal vector, movement vector, and cross product vector
                    normal_vector = find_normal_vector(p_q_points[0], p_q_points[1], p_q_points[2])
                    movement_vector = compute_movement_vector(p_q_points[0], p_q_points[1])
                    cross_product_vector = compute_cross_product_vector(movement_vector, normal_vector)

                    # Calculate the transformation matrix
                    T_base_welding = welding_to_base_matrix(normal_vector, movement_vector, cross_product_vector, welding_point)

                    # Optimize seam tracking in real-time
                    q_i, T_6i = optimized_seam_tracking(T_base_welding, T_6i, q_i, d, a, alpha, T_welding_to_camera, intrinsic_matrix)

                    # Move the robot to the new joint angles
                    robot.move_to_joint_positions(q_i)

        time.sleep(0.1)  # Adjust sleep time for desired control rate

    camera.release()
    cv2.destroyAllWindows()

# Example usage (with mock data for demonstration purposes):
d = [0.1, 0.2, 0.15, 0.1, 0.1, 0.1]
a = [0.5, 0.3, 0.2, 0.1, 0.1, 0.05]
alpha = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, -np.pi/2]
T_welding_to_camera = np.eye(4)  # Example transformation matrix (identity)
intrinsic_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Example intrinsic parameters
initial_joint_angles = [0, 0, 0, 0, 0, 0]  # Example initial joint angles

camera = cv2.VideoCapture(0)  # Open the default camera

robot = OptimizedRobot(d, a, alpha, T_welding_to_camera, intrinsic_matrix)
T_star = np.eye(4)  # Desired end effector pose (example)

real_time_welding(robot, T_star, initial_joint_angles, camera, d, a, alpha, T_welding_to_camera, intrinsic_matrix)
