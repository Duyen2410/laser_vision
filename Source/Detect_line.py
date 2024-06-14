import cv2 as cv
import numpy as np
import para_of_checkerboard as pack
import pro_paths as pp

path = ''

def create_objpoint():
    width = 4
    height = 5
    square_size = 30
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.
    return objp

def Laser_center(img):
    center = np.zeros_like(img)
    # find the center point 
    for x in range(img.cols):
        sum1 = 0.0
        sum2 = 0.0
        roi = np.where(img[:, x] == 255)
        if roi[0].size != 0 :
            for y in roi[0]:
                sum1 += y*img[y][x]
                sum2 += img[y][x]
                center[int(sum1/sum2)][x] =255
        return center
    
def Laser_position(size_of_checker, checkerPath, laserPath, save_cam_para_path):
    print ('get 5 points')
    chessraw = cv.imread(checkerPath)
    chess = cv.cvtColor(chessraw, cv.COLOR_BAYER_BG2GRAY)
    laserraw = cv.imread(laserPath)
    laser = cv.cvtColor(laserraw,cv.COLOR_BAYER_BG2GRAY)
    rows, cols = laser.shape

    cv_file = cv.FileStorage(save_cam_para_path, cv.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("instrinsic").mat()
    dist_matrix = cv_file.getNode('distortion').mat()
    cv_file.release()
    print("read camera parameters success")
    fx = camera_matrix[0][1]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]

    newcameramtx, _ = cv.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (rows, cols), 1, (rows, cols))
    chess_undis = cv.undistort(chess, camera_matrix, dist_matrix, None, newcameramtx)
    laser_undis = cv.undistort(laser, camera_matrix, dist_matrix,  None, newcameramtx )
    ret, corners = cv.findChessboardCorners(chess_undis, size_of_checker, None, cv.CALIB_CB_ADAPTIVE_THRESH)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3000, 0.00001)
    corners_pos1 = cv.cornerSubPix(chess_undis, corners, (11, 11), (-1, -1), criteria=criteria)

    chess_undis_draw = cv.drawChessboardCorners
    if ret:
        objp = create_objpoint()
        retval, rvec_posision, tvec_point = cv.solvePnP(objp, corners_pos1, camera_matrix, dist_matrix)
        rotation_matrix = np.zeros(shape = (3,3))
        cv.Rodrigues(rvec_posision, rotation_matrix)

    blur = cv.GaussianBlur(laser_undis,(7,7),0)
    _, thresh = cv.threshold(blur, 120, 255, cv.THRESH_BINARY)
    closing = thresh
    for _ in range(7):
        closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, np.ones((7,7), np.uint8))
    
    thinned = cv.ximgproc.thinning(closing) 

    cv.imshow('close', closing)
    cv.imshow('thin', thinned)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detec_line():
    checkerPath = path + 'checker_01.jpg'
    laserPath = path + 'laser_01.jpg'
    Laser_position(checkerPath, laserPath)

    