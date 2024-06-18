import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import scipy
import para_of_checkerboard as pack
import pro_paths as pp


#**************************************************************************************
# Hàm load_images: Tải danh sách đường dẫn hình ảnh từ một thư mục cụ thể
# args: đường dẫn tới thư mục chứa hình ảnh (image_dir), tiền tố (image_prefix), định dạng (image_formax)
# return: đường dẫn tới danh sách hình ảnh
#**************************************************************************************
def load_images(image_dir, image_prefix, image_format):
    image_paths = glob.glob(os.path.join(image_dir,f"{image_prefix}*.{image_format}"))
    return image_paths


#**************************************************************************************
# Hàm create_objpoint: Tạo tọa độ các điểm 3D trong không gian thực cho bàn cờ
# args: Kích thước bàn cờ (width, height), kích thước từng ô cờ (square_size)
# return: Tạo độ các điểm trong không gian thực
#**************************************************************************************
def create_objpoint():
    '''
    1. Tạo một mảng hai chiều để chứa các đối tượng trong không gian 3 chiều.
    2. Trong đó hai cột đầu là tọa độ các điểm 2D.
    3. Kích thước thực được tính bằng hệ đơn vị thực.
    '''    
    width = 4
    height = 5
    square_size = 30
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.
    return objp

def load_camera_params(save_camera_params_path_arg):
    cv_file = cv.FileStorage(save_camera_params_path_arg, cv.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("instrinsic").mat()
    dist_matrix = cv_file.getNode("distortion").mat()
    cv_file.release()
    return camera_matrix, dist_matrix

def processing_raw_image(Image_Path_arg):
    Imgraw = cv.imread(Image_Path_arg)
    Img = cv.cvtColor(Imgraw, cv.COLOR_BGR2GRAY)     
    return Img
     
def undistort_images(rows_arg, cols_arg, Img_arg, camera_matrix_arg, dist_matrix_arg):
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(camera_matrix_arg, dist_matrix_arg, (rows_arg, cols_arg), 1, (rows_arg, cols_arg))
    Image_undis = cv.undistort(Img_arg, camera_matrix_arg, dist_matrix_arg, None, newcameramtx)
    return Image_undis

def process_laser_image(laser_undis_arg):
    blur = cv.GaussianBlur(laser_undis_arg,(7,7),0)                 
    _, thresh = cv.threshold(blur, 120, 255, cv.THRESH_BINARY)
    closing = thresh
    for _ in range(7):
        clos = cv.morphologyEx(closing, cv.MORPH_CLOSE, np.ones((7,7),np.uint8))
    thin = cv.ximgproc.thinning(closing)
    return thin, clos   

def LaserCenter(img):
        center = np.zeros_like(img)
        # find the center point
        rows, cols = img.shape
        for x in range(cols):
            sum1 = 0.0
            sum2 = 0.0
            roi = np.where(img[:,x] == 255)
            if roi[0].size != 0:
                for y in roi[0]:
                    sum1 += y * img[y][x]
                    sum2 += img[y][x]
                center[int(sum1/sum2)][x] = 255
        return center


def laser_Position(checker_image_Path, laser_image_Path, camera_mat, dist_mat, i):
    size_of_checker = (4, 5)
    print(i+1)
    # chess_image = processing_raw_image(checker_image_Path)
    laser_image = processing_raw_image(laser_image_Path)
    chess_image = processing_raw_image(checker_image_Path)
    rows,cols = laser_image.shape
    laser_undist = undistort_images(rows, cols,laser_image, camera_mat, dist_mat)
    chess_undist = undistort_images(rows, cols,chess_image, camera_mat, dist_mat)
    ret, corners = cv.findChessboardCorners(chess_undist, size_of_checker, None, cv.CALIB_CB_ADAPTIVE_THRESH)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3000, 0.00001)
    corners_pos1 = cv.cornerSubPix(chess_undist,corners,(11,11),(-1,-1), criteria)
    chess_undis = cv.drawChessboardCorners(chess_undist, size_of_checker, corners_pos1,ret)

    if ret:
        objp = create_objpoint()
        retval, rvec_pos, tvec = cv.solvePnP(objp,corners_pos1, camera_mat, dist_mat)
        rotation_matrix = np.zeros(shape=(3,3))
        cv.Rodrigues(rvec_pos, rotation_matrix)
    thinned, closing = process_laser_image(laser_undist)
    cv.imshow('close', closing)
    cv.imshow('thin', thinned)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thinned, laser_undist, tvec


def extract_laser_point(thinned, fx, fy, cx, cy, rotation_matrix, laser_undis, tvec):
    pointinlaserplane = []
    rows, cols = laser_undis.shape
    line = LaserCenter(thinned)     
    inv = np.linalg.inv(rotation_matrix)
    for i in range(400,rows-200,1):
        for j in range(650,cols-650,1):
            if line[i][j] == 255:
                cv.circle(laser_undis, (j,i), 5, [0,255,0], 2)
                Zc = (tvec[0][0] * inv[2][0] +  tvec[1][0] * inv[2][1] + tvec[2][0] * inv[2][2])/(inv[2][0]/fx*(j-cx) + inv[2][1]/fy*(i-cy) + inv[2][2])
                C = np.array([Zc / fx * (j - cx), Zc / fy * (i - cy), Zc])
                pointinlaserplane.append(C)  

    return pointinlaserplane

def fit_plane_tls(points):
    """
    Fit a plane to a set of points using the Total Least Squares method.

    Parameters:
    points (ndarray): An Nx3 array of points (x, y, z).

    Returns:
    (a, b, c, d): Coefficients of the plane equation ax + by + cz + d = 0.
    """
    # Ensure points is a numpy array
    points = np.asarray(points)
    
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Center the points by subtracting the centroid
    centered_points = points - centroid
    
    # Perform Singular Value Decomposition (SVD)
    _, _, vh = np.linalg.svd(centered_points)
    
    # The normal vector of the plane is the last row of vh
    normal_vector = vh[-1][:]
    print (normal_vector)
    

    
    # Plane equation is given by a*x + b*y + c*z + d = 0
    # Where (a, b, c) is the normal vector and d can be found using the centroid
    a, b, c = normal_vector
    d = -np.dot(normal_vector, centroid)
    
    return a, b, c, d

def plot_plane(a, b, c, d, points=None):
    """
    Plot a plane given by the equation ax + by + cz + d = 0.

    Parameters:
    a, b, c, d (float): Coefficients of the plane equation.
    points (ndarray): Optional Nx3 array of points to plot.
    """
    # Create a meshgrid for the plane
    xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
    zz = (-a * xx - b * yy - d) / c

    # Plotting the plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=100, cstride=100)

    # Plotting the points if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



if __name__ == '__main__':
    pointinlaserplanes = []
    Checkerboard_calib_laser_path = pp.Checkerboard_calib_laser_path
    Laser_position_output_path = pp.Laser_position_output_path
    image_checker_prefix = 'checker_'
    image_laser_prefix = 'laser_'
    image_format = 'jpg'
    checkerPaths = load_images(Checkerboard_calib_laser_path, image_checker_prefix, image_format)
    laserPaths = load_images(Checkerboard_calib_laser_path, image_laser_prefix, image_format)
    save_camera_params_path = pp.save_camera_params_path
    camera_mat, dist_mat = load_camera_params(save_camera_params_path)
    fx = camera_mat[0][0]
    fy = camera_mat[1][1]
    cx = camera_mat[0][2]
    cy = camera_mat[1][2]
    print('read camera parameters success')
    for i, checker_img_path, laser_img_path in zip(range(16), checkerPaths, laserPaths):
        thinned, laser_und, tvec = laser_Position(checker_img_path, laser_img_path, camera_mat, dist_mat,i)
        thinned_Image = Image.fromarray(thinned)
        thinned_Image.save(Laser_position_output_path + f"thinned_{i+1}.png")
        pointlaser = extract_laser_point(thinned, fx, fy, cx, cy, camera_mat,laser_und,tvec)
        pointinlaserplanes.extend(pointlaser)
    
        pointinlaserplane_array = np.array(pointinlaserplanes)

    a, b, c, d =  fit_plane_tls(pointinlaserplane_array)  
    plot_plane(a, b, c, d, points=None)

    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        

    