import cv2 as cv
import numpy as np
import para_of_checkerboard as pack
import pro_paths as pp

path = 'C:/Users/ASUS/Desktop/THINKALPHA/233/NCKH/laser-vision/Calib_data/Checkerboard_calib_laser/'
pointinlaserplane = []

def create_objpoint():
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

def processing_raw_image(checkerPath_arg, laserPath_arg):
    chessraw = cv.imread(checkerPath_arg)
    chess = cv.cvtColor(chessraw, cv.COLOR_BGR2GRAY)   
    laserraw = cv.imread(laserPath_arg)
    laser = cv.cvtColor(laserraw, cv.COLOR_BGR2GRAY)   
    return chess, laser
     
def undistort_images(rows_arg, cols_arg, chess_arg, laser_arg, camera_matrix_arg, dist_matrix_arg):
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(camera_matrix_arg, dist_matrix_arg, (rows_arg, cols_arg), 1, (rows_arg, cols_arg))
    chess_undis = cv.undistort(chess_arg, camera_matrix_arg, dist_matrix_arg, None, newcameramtx)
    laser_undis = cv.undistort(laser_arg, camera_matrix_arg, dist_matrix_arg, None, newcameramtx)
    return chess_undis, laser_undis

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


def find_corners(size_of_checker_arg, camera_mat_arg, dist_mat_arg, chess_undist_arg):
    ret, corners = cv.findChessboardCorners(chess_undist_arg, size_of_checker_arg, None, cv.CALIB_CB_ADAPTIVE_THRESH)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3000, 0.00001)
    corners_pos1 = cv.cornerSubPix(chess_undist_arg,corners,(11,11),(-1,-1), criteria)
    chess_undis = cv.drawChessboardCorners(chess_undist_arg, size_of_checker_arg, corners_pos1,ret)

    if ret:
        objp = create_objpoint()
        retval, rvec_pos, tvec = cv.solvePnP(objp,corners_pos1, camera_mat_arg, dist_mat_arg)
        rotation_matrix = np.zeros(shape=(3,3))
        cv.Rodrigues(rvec_pos, rotation_matrix)
    

def laser_Position(checkerPath, laserPath):
    print('get 5 points')
    size_of_checker = (4, 5)
    chess_image, laser_image = processing_raw_image(checkerPath,laserPath)
    rows,cols = laser_image.shape
    save_camera_params_path = pp.save_camera_params_path
    camera_mat, dist_mat = load_camera_params(save_camera_params_path)
    print('read camera parameters success')
    fx = camera_mat[0][0]
    fy = camera_mat[1][1]
    cx = camera_mat[0][2]
    cy = camera_mat[1][2]

    chess_undist, laser_undist = undistort_images(rows, cols, chess_image, laser_image, camera_mat, dist_mat)
    
    find_corners(size_of_checker, camera_mat, dist_mat, chess_undist)
    

    thinned, closing = process_laser_image(laser_undist)
    cv.imshow('close', closing)
    cv.imshow('thin', thinned)
    cv.waitKey(0)
    cv.destroyAllWindows()


def laser_Calibrate():
    checkerPath = path + 'checker_01.jpg'
    laserPath = path + 'laser_01.jpg'
    laser_Position(checkerPath, laserPath)


if __name__ == '__main__':
    # laser_Position()
    laser_Calibrate()

    