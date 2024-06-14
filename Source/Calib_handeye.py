import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import para_of_checkerboard as pack
import pro_paths as pp

# Hàm load_pos: Để đọc và trích xuất dữ liệu từ một tệp
# args: Đường dẫn tới hình ảnh (path)
# return: Trả về data lấy được dưới dạng ma trận 
def load_pos(path):
    m = []
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    a = cv_file.getNode("K1").mat()
    cv_file.release()
    for i in a:
        m.append(i)
    return m


# Hàm prepare_object_points: Tạo tọa độ các điểm 3D trong không gian thực cho bàn cờ
# args: Kích thước bàn cờ (width, height), kích thước từng ô cờ (square_size)
# return: Tạo độ các điểm trong không gian thực
def create_objpoint(width, height, square_size):
    '''
    1. Tạo một mảng hai chiều để chứa các đối tượng trong không gian 3 chiều.
    2. Trong đó hai cột đầu là tọa độ các điểm 2D.
    3. Kích thước thực được tính bằng hệ đơn vị thực.
    '''
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.
    return objp


# Hàm combine_R_t: Kết hợp ma trận xoay và vector tịnh tiến để tạo thành một ma trận chuyển
# agrs: Ma trận xoay (R), vector tịnh tiến (t)
# return: Ma trận chuyển (h)
def combine_R_t(R, t):
    r_4 = np.array([[0, 0, 0, 1]])
    r_first_2_third = np.concatenate((R, t), axis = 1)
    h = np.concatenate((r_first_2_third, r_4))
    return h


# Hàm load_camera_para: Đọc ma trận thông số nội, ma trận hệ số nhiễu 
# args : Đường dẫn đến tệp lưu ma trậ thông số nội, ma trận hệ số nhiễu (save_camera_params_path)
# return: ma trận thông số nội (camera_matrix), ma trận hệ số nhiễu (dist_matrix)
def load_camera_para(save_camera_params_path):
    cv_file = cv.FileStorage(save_camera_params_path, cv.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("instrinsic").mat()
    dist_matrix = cv_file.getNode("distortion").mat()
    cv_file.release()
    print('read success')
    return camera_matrix, dist_matrix

# Hàm find_corners: Tìm góc checkerboard từ đó xác định hệ số của ma trận thông số ngoại
# args : Kích thước board (width, height), ma trận thông số nội (camera_matrix), ma trận hệ số nhiễu (dist_matrix)
#        Hệ tọa độ các góc trong hệ tọa độ bàn cờ (objp), Đường dẫn thư mục chứa hình ảnh checkerboard (checker_path)
# return: Ma trận thông số ngoại (R_target2cam, t_target2cam)
def find_corners (width, height, camera_matrix, dist_matrix, objp, checker_path):
    '''
    1. Khởi tạo các list lưu trữ các ma trận xoay và các vector tịnh tiến.
    2. Thiết lập điều kiện dừng.
    3. Lấy danh sách các đường dẫn hình ảnh.
    4. Lặp qua từng ảnh.
    5. Đọc và xử lí ảnh.
    6. Tìm góc của board.
        Nếu tìm thấy góc.
            Ma trận xoay.
            Vector tịnh tiến.
    7. Trả về kết quả.
    '''
    R_target2cam = []
    t_target2cam = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3000, 0.00001)
    find_chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH

    images = glob.glob(checker_path +  '*.jpg')
    for frame in images:
        img = cv.imread(frame)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (width, height), None, find_chessboard_flags)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            calib_img = cv.drawChessboardCorners(img, (width, height), corners2, ret)
            # Return the Rotation and the Translation VECTORS that transform a 
            # 3D point expressed in the object coordinate frame to the camera coordinate frame
            retval, rvec, tvec	= cv.solvePnP(objp, corners2, camera_matrix, dist_matrix, flags=cv.SOLVEPNP_ITERATIVE)
            rotation_matrix = np.zeros(shape=(3, 3))
            # Convert a rotation matrix to a rotation vector or vice versa
            cv.Rodrigues(rvec, rotation_matrix)
            R_target2cam.append(rotation_matrix)
            t_target2cam.append(tvec)
            print(frame)
            cv.imshow(frame, calib_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print('fail to find corner')
    return R_target2cam, t_target2cam


# Hàm calibrate_handeye: Tính ma trận chuyển từ camera tới khâu cuối
# args : Ma trận chuyển từ khâu cuối tới base (R_gripper2base, t_gripper2base)
#        Ma trận chuyển từ khâu cuối tới camera-ma trận thông số ngoại (R_target2cam, t_target2cam,)
# return : Ma trận chuyển từ camera tới khâu cuối
def calibrate_handeye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
    # Use Daniilidis due to better accuracy than Tsai
    # Do not use Tsai due to poor accuracy
    R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(R_gripper2base, t_gripper2base, \
                                                       R_target2cam, t_target2cam, method=cv.CALIB_HAND_EYE_DANIILIDIS)
    h_cam2gripper = combine_R_t(R_cam2gripper, t_cam2gripper)
    return h_cam2gripper

# Hàm save_h_cam2gripper: Lưu ma trận h_cam2gripper vào file chỉ định
# args: h_cam2gripper, đường dẫn tới file lưu (save_calib_handeye)
# return: None 
def save_h_cam2gripper(h_cam2gripper, save_calib_handeye):
    cv_file = cv.FileStorage(save_calib_handeye, cv.FILE_STORAGE_WRITE)
    cv_file.write("h_cam2gripper", h_cam2gripper)
    cv_file.release()
    print('save h_cam2gripper success')

def calib_Handeye():
    '''
    1. Tọa độ thực cho các góc checkerboeard trong hệ tọa độ gắn với checkerboard.
    2. Đọc file lấy về ma trận thông số nội và ma trận hệ số nhiễu.
    3. Tìm ma trận thông số ngoại.
    4. Tìm ma trận chuyển từ cam tới khâu cuối. 
    5. Lưu ma trận vừa tìm được vào file chỉ định.
    '''
    checkerboard_size = pack.checkerboard_size
    square_size = pack.square_size
    checker_path = pp.checker_path
    save_camera_params_path = pp.save_camera_params_path
    save_calib_handeye  =  pp.save_calib_handeye
    R_path = pp.R_path 
    t_path = pp.t_path
    R_gripper2base = load_pos(R_path)
    t_gripper2base = load_pos(t_path)
    num_of_image = len(R_gripper2base)
    
    object_point = create_objpoint(*checkerboard_size, square_size)
    cam_mat, dist_mat = load_camera_para(save_camera_params_path)
    R_target2camera, t_target2camera = find_corners (*checkerboard_size, cam_mat, dist_mat, object_point, checker_path)
    h_camera2gripper = calibrate_handeye(R_gripper2base, t_gripper2base, R_target2camera, t_target2camera, save_calib_handeye)  
    save_h_cam2gripper(h_camera2gripper, save_calib_handeye) 

if __name__ == '__main__':
    calib_Handeye()