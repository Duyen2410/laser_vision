import numpy as np
import cv2
import os
import time
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import sys
sys.path.insert(0, 'C:/Users/ASUS/Desktop/THINKALPHA/233/NCKH/laser-vision/Source/DETECTEX')
import para_of_checkerboard as pack
import pro_paths as pp
import re
import random

def load_pos(path):
    m = []
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    a = cv_file.getNode("K1").mat()
    cv_file.release()
    for i in a:
        m.append(i)
    return m

def load_camera_para(save_camera_params_path):
    cv_file = cv.FileStorage(save_camera_params_path, cv.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("instrinsic").mat()
    dist_matrix = cv_file.getNode("distortion").mat()
    cv_file.release()
    print('read success')
    return camera_matrix, dist_matrix

def load_transformation_matrix(path):
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not cv_file.isOpened():
        raise FileNotFoundError(f"Could not open file: {path}")
    
    print("Available nodes in the file:")
    print(cv_file.root().keys())  # Print all available nodes to verify the correct one

    node = cv_file.getNode("h_cam2gripper")
    if node.empty():
        raise ValueError(f"Node 'transformation' not found in the file: {path}")
    
    transformation_matrix = node.mat()
    cv_file.release()
    
    if transformation_matrix.shape == (3, 3):
        transformation_matrix = np.vstack([np.hstack([transformation_matrix, np.array([[0], [0], [0]])]), np.array([0, 0, 0, 1])])
    return transformation_matrix

def undistort_images(rows_arg, cols_arg, Img_arg, camera_matrix_arg, dist_matrix_arg):
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(camera_matrix_arg, dist_matrix_arg, (rows_arg, cols_arg), 1, (rows_arg, cols_arg))
    Image_undis = cv.undistort(Img_arg, camera_matrix_arg, dist_matrix_arg, None, newcameramtx)
    return Image_undis

#def load_images(image_dir, image_prefix, image_format):
    image_paths = glob.glob(os.path.join(image_dir,f"{image_prefix}*.{image_format}"))
    return image_paths

def detect_laser_line(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    laser_line_points = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            laser_line_points.append(((x1, y1), (x2, y2)))
    
    return laser_line_points

def sobel_Y_Image(image):
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))
    return sobel_y

def ransac_Line(x, y):
    count_final = 0
    l = len(x)
    print(l)
    i = 0
    y_min_final = 0
    y_max_final = 0
    while i < 30:
        index_1 = random.randint(0, int(l/2))
        index_2 = random.randint(int(l/2 + 1), l-1)
        if (index_1 != index_2):
            i += 1
            x1 = x[index_1]
            x2 = x[index_2]
            y1 = y[index_1]
            y2 = y[index_2]
            if x1 != x2:
                y_arr = []
                m = (y2-y1)/(x2-x1) 
                c = y1 - (m * x1)   
                count = 0
                for j in range(l):
                    if (j != index_1 and j != index_2):
                        x3 = x[j]
                        y3 = y[j]
                        d = abs((m*x3 - y3 + c))/((m**2 + 1)**0.5)  
                        if d < 2:
                            count += 1
                            y_arr.append(y3)
                if count > count_final:
                    count_final = count
                    m_final = m
                    c_final = c
                    y_min_final = min(y_arr)
                    y_max_final = max(y_arr)
    return [m_final, c_final, y_min_final, y_max_final]

def get_Pos_Line_Laser(image):
   
    center = np.zeros_like(image, dtype=np.uint8)
    
    pos_line_x = []
    pos_line_y = []
    
    for y in range(image.shape[0]):
        sum1 = 0.0
        sum2 = 0.0
        
        roi = np.where(image[y, :] != 0)
        
        if roi[0].size != 0:
            # center[y, roi[0][0]] = 255
            for x in roi[0]:
                sum1 += x
                sum2 += 1
            center[y][round(sum1/sum2)] = 255
            pos_line_x.append(int(sum1/sum2))
            pos_line_y.append(y)
    return [center, pos_line_x, pos_line_y]


def get_Pos_Line_Weld(image):
    
    center = np.zeros_like(image, dtype=np.uint8)
    
    pos_line_x = []
    pos_line_y = []
   
    for x in range(image.shape[1]):
        sum1 = 0.0
        sum2 = 0.0
        roi = np.where(image[:, x] != 0)
        if roi[0].size != 0:
            # center[y, roi[0][0]] = 255
            for y in roi[0]:
                sum1 += y
                sum2 += 1
            center[int(sum1/sum2)][x] = 255
            pos_line_x.append(x)
            pos_line_y.append(int(sum1/sum2))
    return [center, pos_line_x, pos_line_y]

def get_Laser_Line(gray_img):
    
    subImage = gray_img[300:1000, 400:1000]
    
    hist = cv.calcHist([subImage], [0], None, [256], [0,256])
    threshold = 0
    sum_of_points = 0
    for i in range(256):
        sum_of_points += hist[255-i]
        if sum_of_points > 1000:
            threshold = 255 - i -1
            break
    
    ret2, th1 = cv.threshold(subImage, threshold, 255, cv.THRESH_BINARY)
    
    bgr_image = cv.cvtColor(th1, cv.COLOR_GRAY2BGR)
    
    kernel = np.ones((7,7),np.uint8)
    closed_image = cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel)
    
    center, pos_line_x, pos_line_y = get_Pos_Line_Laser(closed_image)
    
    
    m, c, y_min_final, y_max_final = ransac_Line(pos_line_x, pos_line_y)
    
    
    x1 = round((y_min_final - c)/m)
    x2 = round((y_max_final - c)/m)
    roi_x1 = min(x1, x2) - 20
    roi_x2 = max(x1, x2) + 20
    roi_y1 = y_min_final - 5
    roi_y2 = y_max_final + 5
    point1_laser_new = (round((roi_y1-c)/m - roi_x1), 0)
    
    c_new = -m*point1_laser_new[0]
  
    roi = subImage[roi_y1:roi_y2, roi_x1:roi_x2]
    
    max_freq_value = np.argmax(hist)
    
    roi[roi > max_freq_value] = max_freq_value
    
    sobel_y = sobel_Y_Image(roi) 
    hist2 = cv.calcHist([sobel_y], [0], None, [256], [0,256])
    sum_of_points = 0
    for i in range(256):
        sum_of_points += hist2[255-i]
        if sum_of_points > 100:
            threshold = 255 - i -1
            break
    ret2, th2 = cv.threshold(sobel_y, threshold, 255, cv.THRESH_BINARY)
    center2, pos_line_x2, pos_line_y2 = get_Pos_Line_Weld(th2)
    m2, c2, _, __ = ransac_Line(pos_line_x2, pos_line_y2)
    #********************************************
    point_x = round((c2 - c_new) / (m - m2))
    point_y = round(m2 * point_x + c2)
    point_x_new = point_x + 400 + roi_x1
    point_y_new = point_y + 300 + roi_y1
    print([point_x_new, point_y_new])
    #********************************************
    return point_x_new, point_y_new

def load_robot_positions(file_path):
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.split()))
            position = {
                'position': values[:3],
                'orientation':values[3:]
            }
            positions.append(position)
    return positions

# Hàm tính toán ma trận chuyển tiếp từ vị trí robot tới base
def compute_transformation_matrix(position, orientation):
    T = np.eye(4)
    T[:3, 3] = position[0], position[1], position[2]

    # Chuyển đổi góc Euler sang ma trận quay (rotation matrix)
    #roll, pitch, yaw = orientation
    roll, pitch, yaw = 0, 0, 0
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    T[:3, :3] = Rz @ Ry @ Rx
    #T[:3, :3] = Rx @ Ry @ Rz
    return T

def pixel_to_real_world(x, y, fx, fy, cx, cy, a, b, c, d,external_matrix):
    Zc = -d/(a*(x-cx)/fx + b*(y-cy)/fy + c)
    Yc = Zc*(y-cy)/fy
    Xc = Zc*(x-cx)/fx
    point_3d_camera = [Xc, Yc, Zc]
    point_3d_world = np.linalg.inv(external_matrix) @ np.append(point_3d_camera, 1)
    return point_3d_world[:3]


def process_images_and_calculate_joint_angles(fx, fy, cx, cy, a_p, b_p, c_p, d_p, camera_to_welding, intrinsic_matrix, dist_matrix, image_folder, position_file):

    p_list = []
    draw_list = []

    images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))],
                    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

    robot_positions = load_robot_positions(position_file)
    
    for idx, image_path in enumerate(images):
        if idx >= len(robot_positions):
            break
        
        print(f"Processing {image_path} with corresponding robot position index {idx}")
        
        image = cv.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, cols = gray_image.shape
        gray_image = undistort_images(rows, cols, gray_image, intrinsic_matrix, dist_matrix)

        intersection_point = get_Laser_Line(gray_image)
        
        position = robot_positions[idx]['position']
        orientation = robot_positions[idx]['orientation']
        T_i = compute_transformation_matrix(position, orientation)

        if intersection_point is not None:
            matrix = T_i @ camera_to_welding
            external_matrix = np.linalg.inv(matrix)
            welding_point_real_world = pixel_to_real_world(intersection_point[0], intersection_point[1], fx, fy, cx, cy, a_p, b_p, c_p, d_p, external_matrix)
            p_list.append(welding_point_real_world)
            draw_list.append(welding_point_real_world)

    return draw_list

def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]

    ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

if __name__ == "__main__":

    save_camera_params_path = pp.save_camera_params_path
    save_calib_handeye = pp.save_calib_handeye
    intrinsic_matrix, dist_mat = load_camera_para(save_camera_params_path)
    camera_to_welding = load_transformation_matrix(save_calib_handeye)

    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]

    a_p, b_p, c_p, d_p = 42.9397, -0.8313, -1, 2078.2071
    image_folder = "C:/Users/ASUS/Desktop/THINKALPHA/233/NCKH/laser-vision/Scan_data/Scan_data2/"
    position_file = "C:/Users/ASUS/Desktop/THINKALPHA/233/NCKH/laser-vision/Scan_data/Scan_data1/Scan_pos.txt"
    draw_list = process_images_and_calculate_joint_angles(fx, fy, cx, cy, a_p, b_p, c_p, d_p, camera_to_welding, intrinsic_matrix, dist_mat, image_folder, position_file)
    plot_3d_points(draw_list)