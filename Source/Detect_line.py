import cv2 as cv
import numpy as np
from PIL import Image
import glob
import os
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
        for x in range(img.cols):
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
    print(i+1)
    # chess_image = processing_raw_image(checker_image_Path)
    laser_image = processing_raw_image(laser_image_Path)
    rows,cols = laser_image.shape
    laser_undist = undistort_images(rows, cols,laser_image, camera_mat, dist_mat)
    # chess_undist = undistort_images(rows, cols,chess_image, camera_mat, dist_mat)
    thinned, closing = process_laser_image(laser_undist)
    cv.imshow('close', closing)
    cv.imshow('thin', thinned)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thinned
    

if __name__ == '__main__':

    Checkerboard_calib_laser_path = pp.Checkerboard_calib_laser_path
    Laser_position_output_path = pp.Laser_position_output_path
    image_checker_prefix = 'checker_'
    image_laser_prefix = 'laser_'
    image_format = 'jpg'
    checkerPaths = load_images(Checkerboard_calib_laser_path, image_checker_prefix, image_format)
    laserPaths = load_images(Checkerboard_calib_laser_path, image_laser_prefix, image_format)
    save_camera_params_path = pp.save_camera_params_path
    camera_mat, dist_mat = load_camera_params(save_camera_params_path)
    print('read camera parameters success')
    for i, checker_img_path, laser_img_path in zip(range(16), checkerPaths, laserPaths):
        thinned = laser_Position(checker_img_path, laser_img_path, camera_mat, dist_mat,i)
        thinned_Image = Image.fromarray(thinned)
        thinned_Image.save(Laser_position_output_path + f"thinned_{i+1}.png")
        


        

    