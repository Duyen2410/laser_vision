import cv2 as cv
import numpy as np
import glob
import os
import para_of_checkerboard as pack
import pro_paths as pp

#*********************************************************************************************************
# Hàm load_images: Tải danh sách đường dẫn hình ảnh từ một thư mục cụ thể
# args: đường dẫn tới thư mục chứa hình ảnh (image_dir), tiền tố (image_prefix), định dạng (image_formax)
# return: đường dẫn tới danh sách hình ảnh
#*********************************************************************************************************
def load_images(image_dir, image_prefix, image_format):
    image_paths = glob.glob(os.path.join(image_dir,f"{image_prefix}*.{image_format}"))
    return image_paths

#*********************************************************************************************************
# Hàm prepare_object_points: Tạo tọa độ các điểm 3D trong không gian thực cho bàn cờ
# args: Kích thước bàn cờ (width, height), kích thước từng ô cờ (square_size)
# return: Tạo độ các điểm trong không gian thực
#*********************************************************************************************************
def prepare_object_points(width, height, square_size):
    '''
    ////////////////////////////////////////////////
    1. Tạo một mảng hai chiều để chứa các đối tượng trong không gian 3 chiều
    2. Trong đó hai cột đầu là tọa độ các điểm 2D
    3. Kích thước thực được tính bằng hệ đơn vị thực
    ////////////////////////////////////////////////
    '''
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp *= square_size
    return objp

#*********************************************************************************************************
# Hàm extract_corners: Tìm các góc của các ô checkerboard
# args: Kích thước bàn cờ (width, height), đường dẫn tới hình ảnh (image_paths), Tọa độ không gian 3D (objp)
# return: Tọa độ các góc trong không gian (objpoints), tọa độ các góc trong ảnh (imgpoints), ảnh xám (gray)
#*********************************************************************************************************
def extract_corners(width, height, image_paths, objp):
    '''
    ////////////////////////////////////////////////
    1. Khởi tạo điều kiện dừng
    2. Khởi tạo các danh sách lưu trữ điểm
    3. Lặp qua từng hình ảnh
    4. Đọc và xử lí ảnh
    5. Tìm góc của board
        Nếu tìm thấy góc
            Tọa độ các góc trong không gian thực
            Tọa độ các góc trong không gian ảnh
    6. Trả kết quả
    ////////////////////////////////////////////////
    '''
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    find_chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FILTER_QUADS + cv.CALIB_CB_NORMALIZE_IMAGE
    
    objpoints = []
    imgpoints = []

    for frame in image_paths:
        img = cv.imread(frame)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (width, height), None, find_chessboard_flags)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            calib_img = cv.drawChessboardCorners(img, (width, height), corners2, ret)
            print(frame)
            cv.imshow(frame, calib_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("CHECKERBOARD NOT DETECTED!\t---> IMAGE PAIR: ", frame)
    return objpoints, imgpoints, gray

#*********************************************************************************************************
# Hàm calibrate_camera: thực hiện calib camera tìm ma trận nội và ma trận hệ số nhiễu
# args: Tọa độ các góc trong không gian thực (objpoints), tọa độ các góc trong không gian ảnh (imgpoints), ảnh xám (gray)
# return: ma trận thông số nội, ma trận hệ số nhiễu   
#********************************************************************************************************* 
def calibrate_camera(objpoints, imgpoints, gray):
    calibrate_criteria = (cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 500, 0.0001)
    calib_flags = (cv.CALIB_FIX_PRINCIPAL_POINT + cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_RATIONAL_MODEL + cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5)
    ret, inst, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, cv.CALIB_USE_INTRINSIC_GUESS, criteria=calibrate_criteria)
    return inst, dist

#*********************************************************************************************************
# Hàm save_calibration_params: Lưu các ma trận vừa tìm được vào file chỉ định sẵn
# args: Đường dẫn lưu file (save_camera_params_path), ma trận thông số nội (instrinsic), 
# ma trận thông số nhiễu (dist)
# return: None
#*********************************************************************************************************
def save_calibration_params(save_camera_params_path, instrinsic, dist):
    cv_file = cv.FileStorage(save_camera_params_path, cv.FILE_STORAGE_WRITE)
    cv_file.write("instrinsic", instrinsic)
    cv_file.write("distortion", dist)
    cv_file.release()
    print(f'save calib data success to {save_camera_params_path}')

def calibrate():
    '''
    ////////////////////////////////////////////////
    1. Tải danh sách các hình ảnh từ thư mục hình ảnh.
    2. Tạo hệ tọa độ không gian thực (hệ tọa độ bàn cờ) cho các góc bàn cờ.
    3. Tìm tọa độ góc bàn cờ trong hệ tọa độ ảnh.
    4. Tính toán ma trận thông số nội và ma trận hệ số nhiễu.
    5. Lưu kết qủa vào file đã chỉ định. 
    ////////////////////////////////////////////////  
    '''
    print('Start cablirate... ')
    checkerboard_size = pack.checkerboard_size
    square_size = pack.square_size
    img_prefix = 'checkerboard_'
    img_format = 'jpg'
    checker_path = pp.checker_path
    save_camera_params_path = pp.save_camera_params_path
    image_loads_paths = load_images(checker_path, img_prefix, img_format)
    object_point = prepare_object_points(*checkerboard_size, square_size)
    obj_points, image_poins, gray_image  = extract_corners(*(checkerboard_size), image_loads_paths, object_point)
    instrinsic, distortion = calibrate_camera (obj_points, image_poins, gray_image)
    save_calibration_params(save_camera_params_path, instrinsic, distortion)

if __name__ == '__main__':
    calibrate()




