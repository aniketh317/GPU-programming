# RGB To Gray Image Converter
In this project, a coverter to convert RGBs image to a grayscale images using CUDA programming has been implemented. The project is implemented using CUDA programming and the code is written in C++.

## Details
Launch Cuda Kernel with a 2D block of threads (16,16) and a 2D grid of blocks. The kernel will convert the RGB image to a grayscale image. The RGB image is converted to a 1D array, copied to device. On device, grayscale pixel values are calculated and stored in a 1D array. The grayscale image is then copied back to host and saved as a grayscale image.
