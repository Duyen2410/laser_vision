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
import time
import sys
import numpy as np
from numba import cuda, float32, njit
sys.path.insert(0, '/content/gdrive/MyDrive/COLAB/THINKALPHA/233/NCKH/laser-vision/Source/thunghiem/RANSAC')
import ransac3ex as ransac


def load_images(image_dir, image_prefix, image_format):
    image_paths = glob.glob(os.path.join(image_dir,f"{image_prefix}*.{image_format}"))
    return image_paths


def create_objpoint():
     
    width = 4
    height = 5
    square_size = 30
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size 
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


@cuda.jit
def find_center(img, center):
    y, x = cuda.grid(2)
    rows, cols = img.shape
    if x < cols and y < rows:
        if img[y, x] == 255:
            sum1 = 0.0
            sum2 = 0.0
            for i in range(rows):
                if img[i, x] == 255:
                    sum1 += i * img[i, x]
                    sum2 += img[i, x]
            if sum2 != 0:
                center[int(sum1 / sum2), x] = 255

def LaserCenter(img):
    rows, cols = img.shape
    center = np.zeros_like(img)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (rows + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    find_center[blocks_per_grid, threads_per_block](img, center)

    return center



def laser_Position(checker_image_Path, laser_image_Path, camera_mat, dist_mat, i):
    size_of_checker = (4, 5)
    print(i+1)
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
    
    return thinned, laser_undist, tvec



@cuda.jit
def find_laser_points(thinned, fx, fy, cx, cy, inv, tvec, laser_undis, pointinlaserplane):
    i, j = cuda.grid(2)
    rows, cols = thinned.shape
    if 400 <= i < rows - 200 and 650 <= j < cols - 650:
        if thinned[i, j] == 255:
            Zc = (tvec[0][0] * inv[2, 0] + tvec[1][0] * inv[2, 1] + tvec[2][0] * inv[2, 2]) / ((inv[2, 0] / fx) * (j - cx) + (inv[2, 1] / fy) * (i - cy) + inv[2, 2])
            Cx = Zc / fx * (j - cx)
            Cy = Zc / fy * (i - cy)
            pointinlaserplane[i, j, 0] = Cx
            pointinlaserplane[i, j, 1] = Cy
            pointinlaserplane[i, j, 2] = Zc
            cuda.atomic.add(pointinlaserplane, (i, j, 3), 1)

def extract_laser_point(thinned, fx, fy, cx, cy, rotation_matrix, laser_undis, tvec):
    rows, cols = thinned.shape
    line = LaserCenter(thinned)
    inv = np.linalg.inv(rotation_matrix)
    pointinlaserplane = np.zeros((rows, cols, 4), dtype=np.float32)

    # Configure the blocks
    threads_per_block = (16, 16)
    blocks_per_grid_x = (rows + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (cols + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    find_laser_points[blocks_per_grid, threads_per_block](line, fx, fy, cx, cy, inv, tvec, laser_undis, pointinlaserplane)

    valid_points = pointinlaserplane[:, :, 3] > 0
    points = pointinlaserplane[valid_points][:, :3]

    return points



def plot_plane(xs, ys, zs, fit):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = (fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2])
    ax.plot_wireframe(X,Y,Z, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def find_corners (width, height, camera_matrix, dist_matrix, objp, checker_img_path):

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3000, 0.00001)
    find_chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH
    img = cv.imread(checker_img_path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (width, height), None, find_chessboard_flags)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        calib_img = cv.drawChessboardCorners(img, (width, height), corners2, ret)
        retval, rvec, tvec	= cv.solvePnP(objp, corners2, camera_matrix, dist_matrix, flags=cv.SOLVEPNP_ITERATIVE)
        rotation_matrix = np.zeros(shape=(3, 3))
        cv.Rodrigues(rvec, rotation_matrix)
    else:
        print('fail to find corner')
    return rotation_matrix, tvec

if __name__ == '__main__':
    
    checkerboard_size = pack.checkerboard_size
    square_size = pack.square_size
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
        start_time = time.time()
        thinned, laser_und, tvec = laser_Position(checker_img_path, laser_img_path, camera_mat, dist_mat,i)
        thinned_Image = Image.fromarray(thinned)
        thinned_Image.save(Laser_position_output_path + f"thinned_{i+1}.png")
        object_point = create_objpoint()
        rotation_mat, tvec2 = find_corners (*checkerboard_size, camera_mat, dist_mat, object_point, checker_img_path)
        pointlaser = extract_laser_point(thinned, fx, fy, cx, cy, rotation_mat ,laser_und, tvec2)
        pointinlaserplanes.extend(pointlaser)
        pointinlaserplane_array = np.array(pointinlaserplanes)
        ransac_time = time.time() - start_time
    x, y, z, bestFit, _, max_inliers = ransac.ransac_plane_fitting(pointinlaserplane_array)
    plot_plane(x, y, z, bestFit)
    a, b, c = bestFit
    mse_ransac, r_adj_sq, mae_ransac = ransac.evaluate_models(pointinlaserplane_array, bestFit)
    print(f"Phương trình mặt phẳng từ RANSAC tự triển khai: z = {a:.4f} * x + {b:.4f} * y + {c:.4f}")
    print(f"Thời gian Calib_laser với RANSAC: {ransac_time:.4f} giây")
    print(f"MSE của RANSAC tự triển khai: {mse_ransac:.4f}")
    print(f"R^2 : {r_adj_sq:.4f}")
    print(f"MAE: {mae_ransac:.4f}")
