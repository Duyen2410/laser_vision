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
import sys
import time
sys.path.insert(0, 'C:/Users/ASUS/Desktop/THINKALPHA/233/NCKH/laser-vision/Source/thunghiem/HOUGH')
import RHTBex as HT


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
    ///////////////////////////////////////////////////////////////////////////////////
    1. Tạo một mảng hai chiều để chứa các đối tượng trong không gian 3 chiều.  
    2. Trong đó hai cột đầu là tọa độ các điểm 2D.
    3. Kích thước thực được tính bằng hệ đơn vị thực.
    ///////////////////////////////////////////////////////////////////////////////////
    '''    
    width = 4
    height = 5
    square_size = 30
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.
    return objp


#**************************************************************************************
# Hàm load_camera_params: Tải các tham số camera từ tệp lưu trữ
# args: Đường dẫn tới tệp chứa các tham số camera được lưu trữ (save_camera_params_path_arg)
# return: Các tham số camera từ tệp-ma trận thông số nội, ma trận hệ số nhiễu (camera_matrix, dist_matrix)
#**************************************************************************************
def load_camera_params(save_camera_params_path_arg):
    cv_file = cv.FileStorage(save_camera_params_path_arg, cv.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("instrinsic").mat()
    dist_matrix = cv_file.getNode("distortion").mat()
    cv_file.release()
    return camera_matrix, dist_matrix


#**************************************************************************************
# Hàm load_images: Xử lí ảnh thô bằng cách chuyển nó sang ảnh xám
# args: Đường dẫn tới tệp hình ảnh cần xử lí
# return: Ảnh xám (Img)
#**************************************************************************************
def processing_raw_image(Image_Path_arg):
    Imgraw = cv.imread(Image_Path_arg)
    Img = cv.cvtColor(Imgraw, cv.COLOR_BGR2GRAY)     
    return Img


#**************************************************************************************
# Hàm load_images: Hiệu chỉnh hình ảnh bị nhiễu bằng cách sử dụng sử dụng các tham số camera
# args: Kích thước ảnh (rows_arg, cols_arg), Hình ảnh bị nhiễu (Img_arg), Ma trận thông số nội (camera_matrix_arg), Ma trận hệ số nhiễu (dist_matrix_arg) 
# return: Hình ảnh đã loại bỏ hệ số nhiễu (Image_undis)
#**************************************************************************************
def undistort_images(rows_arg, cols_arg, Img_arg, camera_matrix_arg, dist_matrix_arg):
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(camera_matrix_arg, dist_matrix_arg, (rows_arg, cols_arg), 1, (rows_arg, cols_arg))
    Image_undis = cv.undistort(Img_arg, camera_matrix_arg, dist_matrix_arg, None, newcameramtx)
    return Image_undis

#**************************************************************************************
# Hàm process_laser_image: Xử lí ảnh 
# args: Hình ảnh đã loại hệ số nhiễu (laser_undis_arg)
# return: Ảnh đã xử lí closing(clos), ảnh đã xử lí thinned(thin)
#**************************************************************************************

def process_laser_image(laser_undis_arg):
    '''
    ///////////////////////////////////////////////////////////////////////////////////
    1. Làm mờ ảnh bằng lọc Gauss.  
    2. Xử lí ảnh bằng Threshold.
    3. Xử lí closing.
    4. Xử lí thinned.
    ///////////////////////////////////////////////////////////////////////////////////
    '''
    blur = cv.GaussianBlur(laser_undis_arg,(7,7),0)                 
    _, thresh = cv.threshold(blur, 120, 255, cv.THRESH_BINARY)
    closing = thresh
    for _ in range(7):
        clos = cv.morphologyEx(closing, cv.MORPH_CLOSE, np.ones((7,7),np.uint8))
    thin = cv.ximgproc.thinning(closing)
    return thin, clos   


#**************************************************************************************
# Hàm LaserCenter: Tìm tâm đường laser.
# args: Hình ảnh cần xử lí (img).
# return: Tọa độ tâm đường laser (center)
#**************************************************************************************
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


#**************************************************************************************
# Hàm laser_Position: Tách line (giả sử đường laser là màu trắng)
# args: đường dẫn tới tệp ảnh checkerboard (chess_image_Path), đường dẫn tới tệp ảnh laser (laser_image_Path)
# ma trận thông số nội (camera_mat), ma trận hệ số nhiễu (dist_mat)
# return: ảnh xử lí thinned (thinned), hình ảnh laser đã xử lí (laser_undist), vecto tịnh tiến (tvec)
#**************************************************************************************
def laser_Position(checker_image_Path, laser_image_Path, camera_mat, dist_mat, i):
    '''
    ///////////////////////////////////////////////////////////////////////////////////
    1. Định nghĩa kích thước bàn cờ.
    2. Đọc và xử lí ảnh thô.
    3. Hiệu chỉnh loại bỏ nhiễu hình ảnh.
    4. Xác định tiêu chí để tìm kiếm chính xác góc.
    5. Vẽ các góc bàn cờ trên ảnh.
    6. Nếu tìm thấy bàn cờ, tính toán các ma trận chuyển.
    7. Xử lí ảnh laser closing, thinned.
    ///////////////////////////////////////////////////////////////////////////////////
    '''
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
    #cv.imshow('close', closing)
    #cv.imshow('thin', thinned)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return thinned, laser_undist, tvec

#**************************************************************************************
# Hàm extract_laser_point: Tìm tọa độ các điểm trên mặt phẳng laser trong hệ tọa độ camera
# args: Hình ảnh đã xử lí thinned (thinned), các hệ số của ma trận hệ số nhiễu (fx, fy, cx, cy)
# ma trận xoay từ hệ tọa độ thực sang hệ tọa độ camera (rotation_matrix), 
# Hình ảnh đã xử loại nhiễu (laser_undis), vector tịnh tiến (tvec)
# return: Danh sách điểm trên mặt phẳng laser (pointinlaserplane)
#**************************************************************************************
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

#**************************************************************************************
# Hàm load_images: Tải danh sách đường dẫn hình ảnh từ một thư mục
# args: đường dẫn tới thư mục chứa hình ảnh (image_dir), tiền tố (image_prefix), định dạng (image_formax)
# return: đường dẫn tới danh sách hình ảnh
#**************************************************************************************
def fit_plane_tls(points):
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    print("solution:")
    print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors:")
    print(errors)
    print("residual:")
    print(residual)
    return xs, ys, zs, fit

#**************************************************************************************
# Hàm plot_plane: Vẽ đồ thị
# args: Tọa độ các điểm trên mặt phẳng laser (xs, ys, zs), hệ số mặt phẳng fit (fit)
# return: None
#**************************************************************************************
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


if __name__ == '__main__':
    '''
    ///////////////////////////////////////////////////////////////////////////////////
    1. Tải ma trận thông số nội, ma hệ số nhiễu.
    2. Quét qua từng ảnh checkerboard đồng thời với ảnh laser tương ứng.
    3. Tách line.
    4. Lưu ảnh vào tệp.
    5. Tìm tập tọa độ điểm trên mặt phẳng laser.
    6. Tìm mặt phẳng laser.
    7. Vẽ đồ thị.
    ///////////////////////////////////////////////////////////////////////////////////
    '''
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
    print(pointinlaserplane_array)
    start_time = time.time()
    x, y, z, fix_pl = HT.detect_best_plane(pointinlaserplane_array, k_max_in = 100, n_in = 0.8, theta_res=0.05, phi_res=0.05, rho_res=0.05)
    hough_time = time.time() - start_time
    plot_plane(x, y, z, fix_pl)
    a, b, c, d  = fix_pl
    # Đánh giá các mô hình
    mse_hough = HT.evaluate_models(pointinlaserplane_array, fix_pl)
    print(f"Phương trình mặt phẳng từ HOUGH tự triển khai: z = {-a/c:.4f} * x + {-b/c:.4f} * y + {-d/c:.4f}")
    print(f"Thời gian thực thi với HOUGH tự triển khai: {hough_time:.4f} giây")
    print(f"MSE của Hough transform: {mse_hough:.4f}")
    



        

    