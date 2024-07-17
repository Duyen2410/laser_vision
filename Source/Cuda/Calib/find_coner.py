import cv2 as cv
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Device function để phát hiện góc bàn cờ
module = SourceModule("""
__device__ float computeCornerResponse(unsigned char* image, int width, int x, int y) {
    // Tính gradient (Sobel)
    float dx = (image[y * width + (x + 1)] - image[y * width + (x - 1)]);
    float dy = (image[(y + 1) * width + x] - image[(y - 1) * width + x]);
    return dx * dx + dy * dy;
}

__device__ bool findChessboardCorners_cuda(unsigned char* image, int width, int height, 
                                            int pattern_width, int pattern_height, 
                                            float* corners, int* cornerCount) {
    int count = 0;
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float response = computeCornerResponse(image, width, x, y);
            
            // Ngưỡng phát hiện góc
            if (response > 1000) {
                if (count < pattern_width * pattern_height) {
                    int index = atomicAdd(cornerCount, 1);
                    corners[index * 2] = (float)x;
                    corners[index * 2 + 1] = (float)y;
                    count++;
                }
            }
        }
    }
    
    return count == (pattern_width * pattern_height);
}

__global__ void detectCornersKernel(unsigned char* images, int width, int height, 
                                     int pattern_width, int pattern_height, 
                                     float* corners, int* cornerCount, 
                                     bool* found, int num_images) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Xác định chỉ số hình ảnh
    if (idx >= num_images) return;  // Kiểm tra chỉ số hợp lệ

    unsigned char* image = &images[idx * width * height];  // Lấy con trỏ đến hình ảnh hiện tại
    int local_cornerCount = 0;
    bool local_found = findChessboardCorners_cuda(image, width, height, pattern_width, pattern_height, 
                                                   &corners[idx * pattern_width * pattern_height * 2], 
                                                   &local_cornerCount);

    cornerCount[idx] = local_cornerCount;  // Lưu số lượng góc tìm thấy
    found[idx] = local_found;  // Lưu trạng thái tìm thấy
}
""")

def extract_corners(width, height, image_paths, objp):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern_width, pattern_height = width, height

    objpoints = []
    imgpoints = []

    num_images = len(image_paths)
    
    # Đọc và xử lý tất cả các hình ảnh
    images = []
    for frame in image_paths:
        img = cv.imread(frame)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        images.append(gray.flatten().astype(np.uint8))  # Thêm ảnh vào danh sách

    # Chuyển đổi danh sách ảnh thành mảng 1D
    images_flat = np.concatenate(images).astype(np.uint8)

    # Khởi tạo mảng cho các góc và trạng thái tìm thấy
    max_corners = pattern_width * pattern_height
    corners = np.zeros((num_images, max_corners * 2), dtype=np.float32)  # Mảng góc
    corner_count = np.zeros((num_images,), dtype=np.int32)  # Mảng số lượng góc tìm thấy
    found = np.zeros((num_images,), dtype=np.bool)  # Mảng trạng thái tìm thấy

    # Tải dữ liệu lên GPU
    images_gpu = cuda.mem_alloc(images_flat.nbytes)
    corners_gpu = cuda.mem_alloc(corners.nbytes)
    count_gpu = cuda.mem_alloc(corner_count.nbytes)
    found_gpu = cuda.mem_alloc(found.nbytes)

    cuda.memcpy_htod(images_gpu, images_flat)
    cuda.memcpy_htod(corners_gpu, corners)
    cuda.memcpy_htod(count_gpu, corner_count)
    cuda.memcpy_htod(found_gpu, found)

    # Gọi kernel
    block_size = 32
    grid_size = (num_images + block_size - 1) // block_size  # Số lượng khối

    kernel = module.get_function("detectCornersKernel")
    kernel(images_gpu, np.int32(width), np.int32(height), 
           np.int32(pattern_width), np.int32(pattern_height), 
           corners_gpu, count_gpu, found_gpu, 
           np.int32(num_images), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Sao chép kết quả về CPU
    cuda.memcpy_dtoh(corners, corners_gpu)
    cuda.memcpy_dtoh(corner_count, count_gpu)
    cuda.memcpy_dtoh(found, found_gpu)

    # Xử lý kết quả
    for i in range(num_images):
        if found[i]:
            objpoints.append(objp)
            imgpoints.append(corners[i][:corner_count[i] * 2].reshape(-1, 2))
        else:
            print(f"CHECKERBOARD NOT DETECTED! ---> IMAGE PAIR: {image_paths[i]}")

    return objpoints, imgpoints
