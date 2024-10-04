# RGB To Gray Image Converter
In this project, a simple Linear Blur to Blur an image using CUDA programming has been implemented. The project is implemented using CUDA programming and the code is written in C++.

## Details
Launch Cuda Kernel with a 2D block of threads (16,16) and a 2D grid of blocks. The kernel will apply a simple Linear Blur taking average of the current pixel, pixel on the left and pixel on the right. The RGB image is converted to a 1D array, and is unified memory(address spaec is shared and copy is done by CUDA runtime whenever required). For the purpose of averaging, a shared memory copy of the block is created. The shared memory copy is used to calculate the average of the pixel, pixel on the left and pixel on the right. Shared memory is a faster way for threads in a block to share data and only threads in the block can access that piece of shared memory. Block level synchronization is used.
