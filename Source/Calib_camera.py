import cv2 as cv
import numpy as np
import glob
import os
import para_of_checkerboard as pack
import pro_paths as pp

def load_images(image_dir, image_prefix, image_format):
    image_paths = glob.glob(os.path.join(image_dir,f"{image_prefix}*.{image_format}"))
    return image_paths

def prepare_object_points(width, height, square_size):
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp *= square_size
    return objp

def extract_corners(width, height, image_paths, objp):
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
    
def calibrate_camera(objpoints, imgpoints, gray):
    calibrate_criteria = (cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 500, 0.0001)
    calib_flags = (cv.CALIB_FIX_PRINCIPAL_POINT + cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_RATIONAL_MODEL + cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5)
    ret, inst, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, cv.CALIB_USE_INTRINSIC_GUESS, criteria=calibrate_criteria)
    return inst, dist

def save_calibration_params(save_camera_params_path, instrinsic, dist):
    cv_file = cv.FileStorage(save_camera_params_path, cv.FILE_STORAGE_WRITE)
    cv_file.write("instrinsic", instrinsic)
    cv_file.write("distortion", dist)
    cv_file.release()
    print(f'save calib data success to {save_camera_params_path}')

def calibrate():
    print('Start cablirate... ')
    # Khai bao thuoc tinh anh
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




