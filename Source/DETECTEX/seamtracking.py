import numpy as np
import cv2
import time

def dh_matrix(theta, d, a, alpha):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(joint_angles, d, a, alpha):
    T = np.eye(4)
    T_list = [T]
    for theta, d_i, a_i, alpha_i in zip(joint_angles, d, a, alpha):
        T_i = dh_matrix(theta, d_i, a_i, alpha_i)
        T = T @ T_i
        T_list.append(T)
    return T, T_list

def jacobian(joint_angles, d, a, alpha):
    _, T_list = forward_kinematics(joint_angles, d, a, alpha)
    J = np.zeros((6, len(joint_angles)))

    end_effector_pos = T_list[-1][:3, 3]

    for i in range(len(joint_angles)):
        z_i = T_list[i][:3, 2]
        o_i = T_list[i][:3, 3]

        J[:3, i] = np.cross(z_i, end_effector_pos - o_i)
        J[3:, i] = z_i

    return J

def inverse_jacobian(J):
    try:
        J_inv = np.linalg.pinv(J, rcond=1e-4)
    except np.linalg.LinAlgError:
        J_inv = np.linalg.pinv(J)
    return J_inv

def detect_joint_edge(image):
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

def find_knee_points(edge_points):
    knee_points = []

    for i in range(1, len(edge_points) - 1):
        (x1, y1), (x2, y2) = edge_points[i]
        (x3, y3), (x4, y4) = edge_points[i + 1]

        slope1 = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
        slope2 = (y4 - y3) / (x4 - x3) if x4 != x3 else np.inf

        if np.abs(slope1 - slope2) > 0.1:
            knee_points.append((x2, y2))
    return knee_points

def pixel_to_real_world(x, y, intrinsic_matrix, external_matrix):
    point_homogeneous = np.array([x, y, 1])
    point_3d_camera = np.linalg.inv(intrinsic_matrix) @ point_homogeneous
    point_3d_world = external_matrix @ np.append(point_3d_camera, 1)
    return point_3d_world[:3]

def find_p_q_points(knee_points, intrinsic_matrix, external_matrix, offset=5):
    p_q_points = []

    for (x, y) in knee_points:
        direction_vector = np.array([y, -x])
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)

        point_real_world = pixel_to_real_world(x, y, intrinsic_matrix, external_matrix)

        p_real_world = point_real_world + offset * normalized_vector

        p_q_points.append(p_real_world)

    return p_q_points

def find_welding_point(u_real_world, v_real_world):
    welding_point = (u_real_world + v_real_world) / 2
    return welding_point

def find_normal_vector(p, q, r):
    pq = q - p
    pr = r - p

    normal_vector = np.cross(pq, pr)

    if normal_vector[2] > 0:
        normal_vector = -normal_vector

    normal_vector = normal_vector / np.linalg.norm(normal_vector)

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

def calculate_joint(d, a, alpha, camera_to_welding, intrinsic_matrix, T_i_plus_1, joint_angles):
    T_i, T_list = forward_kinematics(joint_angles, d, a, alpha)
    J = jacobian(joint_angles, d, a, alpha)
    inv_J = inverse_jacobian(J)
    q_i = np.array(joint_angles)
    camera = cv2.VideoCapture(0)
    p_list = []

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = camera.read()
        if ret:
            edges = detect_joint_edge(frame)
            if edges:
                knee_points = find_knee_points(edges)
                if len(knee_points) >= 2:
                    external_matrix = camera_to_welding @ T_i
                    u_real_world = pixel_to_real_world(knee_points[0][0], knee_points[0][1], intrinsic_matrix, external_matrix)
                    v_real_world = pixel_to_real_world(knee_points[1][0], knee_points[1][1], intrinsic_matrix, external_matrix)
                    
                    p_real_world, q_real_world = find_p_q_points(knee_points, intrinsic_matrix, external_matrix, offset=5)
                    
                    p = find_welding_point(u_real_world, v_real_world)
                    p_list.append([p, p_real_world, q_real_world])
                    
                    if len(p_list) > 1:
                        normal_vector = find_normal_vector(p_list[-2][0], p_list[-2][1], p_list[-2][2])
                        movement_vector = compute_movement_vector(p_list[-2][0], p_list[-1][0])
                        cross_vector = compute_cross_product_vector(movement_vector, normal_vector)
                        T = welding_to_base_matrix(normal_vector, movement_vector, cross_vector, p_list[-1][0])
                        
                        dT6 = T[:3, 3] - T_i[:3, 3]  # Only use the translation part
                        dq_i_plus_1 = inv_J @ dT6
                        q_i_plus_1 = q_i + dq_i_plus_1
                        T_i, _ = forward_kinematics(q_i_plus_1, d, a, alpha)
                        q_i = q_i_plus_1
                        time.sleep(0.1)  # Small delay for smoother movement
                else:
                    print("Knee points not detected properly")
        else:
            print("Error in capturing frame.")
            break

    camera.release()
    cv2.destroyAllWindows()
    return q_i

# Example usage:
d = [0.5, 0.2, 0.3]  # DH parameters (example)
a = [0.1, 0.4, 0.2]
alpha = [np.pi/2, 0, -np.pi/2]
camera_to_welding = np.eye(4)  # Example matrix
intrinsic_matrix = np.eye(3)  # Example intrinsic matrix
T_i_plus_1 = np.eye(4)  # Example transformation matrix

joint_angles = [0, 0, 0]  # Initial joint angles (example)
final_joint_angles = calculate_joint(d, a, alpha, camera_to_welding, intrinsic_matrix, T_i_plus_1, joint_angles)
