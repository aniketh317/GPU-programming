/*This handles the block edge case too*/
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "simpleLinearBlurFilter.hpp"

/*
 * CUDA Kernel Device code
 *
 */

__global__ void applySimpleLinearBlurFilter(uchar *r, uchar *g, uchar *b)
{
    // Consider using shared memory for the purpose of keeping the original input values
    // You can also use a constant array for handling edge cases or applying a custom filter

    __shared__ uchar r_old[16][16], g_old[16][16], b_old[16][16];

    int y = blockDim.x*blockIdx.x+threadIdx.x;
    int x = blockDim.y*blockIdx.y+threadIdx.y;

    if(x<d_columns && y<d_rows)
    {
        // When using shared memory you should store pixel values relevant to the current thread in a variable
        r_old[threadIdx.x][threadIdx.y] = r[d_columns*y+x];
        g_old[threadIdx.x][threadIdx.y] = g[d_columns*y+x];
        b_old[threadIdx.x][threadIdx.y] = b[d_columns*y+x];
    }
        // sync threads so that you can alter RGB values without causing race condition
    __syncthreads();
    // Apply a simple filter that averages the RGB values to the left and right of the pixel at the current thread id location
    if(((threadIdx.y)%16)!=0 && x<d_columns-1 && ((threadIdx.y+1)%16)!=0)
    {
        r[d_columns*y+x] = ((int)r_old[threadIdx.x][threadIdx.y-1]+(int)r_old[threadIdx.x][threadIdx.y]+(int)r_old[threadIdx.x][threadIdx.y+1])/3;
        g[d_columns*y+x] = ((int)g_old[threadIdx.x][threadIdx.y-1]+(int)g_old[threadIdx.x][threadIdx.y]+(int)g_old[threadIdx.x][threadIdx.y+1])/3;
        b[d_columns*y+x] = ((int)b_old[threadIdx.x][threadIdx.y-1]+(int)b_old[threadIdx.x][threadIdx.y]+(int)b_old[threadIdx.x][threadIdx.y+1])/3;
    }
    // Another area for improvement is handling when the current thread is at the let or right edge of the imput image
    else if(x>0 && x<d_columns-1 && (threadIdx.y+1)%16==0 && y<d_rows)
    {
        int id = blockIdx.y;
        //When pixel at right edge of the block
        r[d_columns*y+x] = ((int)r_old[threadIdx.x][threadIdx.y-1]+(int)r_old[threadIdx.x][threadIdx.y]+(int)r_e_even[y][id])/3;
        g[d_columns*y+x] = ((int)g_old[threadIdx.x][threadIdx.y-1]+(int)g_old[threadIdx.x][threadIdx.y]+(int)g_e_even[y][id])/3;
        b[d_columns*y+x] = ((int)b_old[threadIdx.x][threadIdx.y-1]+(int)b_old[threadIdx.x][threadIdx.y]+(int)b_e_even[y][id])/3;
    }
    else if(x>0 && x<d_columns-1 && (threadIdx.y)%16==0 && y<d_rows)
    {
        int id = blockIdx.y;
        //When pixel at left edge of the block
        r[d_columns*y+x] = ((int)r_old[threadIdx.x][threadIdx.y]+(int)r_old[threadIdx.x][threadIdx.y+1]+(int)r_e_odd[y][id])/3;
        g[d_columns*y+x] = ((int)g_old[threadIdx.x][threadIdx.y]+(int)g_old[threadIdx.x][threadIdx.y+1]+(int)g_e_odd[y][id])/3;
        b[d_columns*y+x] = ((int)b_old[threadIdx.x][threadIdx.y]+(int)b_old[threadIdx.x][threadIdx.y+1]+(int)b_e_odd[y][id])/3;
    }
}

__host__ float compareColorImages(uchar *r0, uchar *g0, uchar *b0, uchar *r1, uchar *g1, uchar *b1, int rows, int columns)
{
    cout << "Comparing actual and test pixel arrays\n";
    int numImagePixels = rows * columns;
    int imagePixelDifference = 0.0;

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            uchar image0R = r0[r*columns+c];
            uchar image0G = g0[r*columns+c];
            uchar image0B = b0[r*columns+c];
            uchar image1R = r1[r*columns+c];
            uchar image1G = g1[r*columns+c];
            uchar image1B = b1[r*columns+c];
            imagePixelDifference += ((abs(image0R - image1R) + abs(image0G - image1G) + abs(image0B - image1B))/3);
        }
    }

    float meanImagePixelDifference = imagePixelDifference / numImagePixels;
    float scaledMeanDifferencePercentage = (meanImagePixelDifference / 255);
    printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n", meanImagePixelDifference, scaledMeanDifferencePercentage);
    return scaledMeanDifferencePercentage;
}

__host__ void allocateDeviceMemory(int rows, int columns)
{

    //Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__host__ void executeKernel(uchar *r, uchar *g, uchar *b, int rows, int columns, int threadsPerBlock)
{
    cout << "Executing kernel\n";
    //Launch the convert CUDA Kernel

    int gridDimx = (columns>>4)+1; //Number of blocks along x
    int gridDimy = (rows>>4)+1; //Number of blocks along y

    dim3 grid(gridDimy, gridDimx); //grid dimension
    dim3 block(16, 16); //block dimension
    
    uchar **hr_e, **hg_e, **hb_e;
    
    uchar **hr_e_host, **hg_e_host, **hb_e_host;
    hr_e_host = (uchar **)malloc(rows*sizeof(uchar*));
    hg_e_host = (uchar **)malloc(rows*sizeof(uchar*));
    hb_e_host = (uchar **)malloc(rows*sizeof(uchar*));

    /*For odd allocation*/
    cudaMalloc(&hr_e, rows*sizeof(uchar*));
    cudaMalloc(&hg_e, rows*sizeof(uchar*));
    cudaMalloc(&hb_e, rows*sizeof(uchar*));
    for(int i=0;i<rows;i++)
    {
        cudaMalloc(&hr_e_host[i], (gridDimx)*sizeof(uchar));
        cudaMalloc(&hg_e_host[i], (gridDimx)*sizeof(uchar));
        cudaMalloc(&hb_e_host[i], (gridDimx)*sizeof(uchar));
        uchar *temp_r, *temp_g, *temp_b;
        temp_r = (uchar *)malloc((gridDimx)*sizeof(uchar));
        temp_g = (uchar *)malloc((gridDimx)*sizeof(uchar));
        temp_b = (uchar *)malloc((gridDimx)*sizeof(uchar));
        for(int j=1;j<=gridDimx;j++)
        {
            int x = 16*j-1;
            if(x<columns)
            {
                temp_r[j-1] = r[i*columns+x];
                temp_g[j-1] = g[i*columns+x];
                temp_b[j-1] = b[i*columns+x];
            }
        }
        cudaMemcpy(hr_e_host[i], temp_r, gridDimx*sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(hg_e_host[i], temp_g, gridDimx*sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(hb_e_host[i], temp_b, gridDimx*sizeof(uchar), cudaMemcpyHostToDevice);
        free(temp_r);
        free(temp_g);
        free(temp_b);
    }

    cudaMemcpy(hr_e, hr_e_host, rows * sizeof(uchar*), cudaMemcpyHostToDevice);
    cudaMemcpy(hg_e, hg_e_host, rows * sizeof(uchar*), cudaMemcpyHostToDevice);
    cudaMemcpy(hb_e, hb_e_host, rows * sizeof(uchar*), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(r_e_odd, &hr_e, sizeof(uchar **));
    cudaMemcpyToSymbol(g_e_odd, &hg_e, sizeof(uchar **));
    cudaMemcpyToSymbol(b_e_odd, &hb_e, sizeof(uchar **));

    /*For even allocation*/
    cudaMalloc(&hr_e, rows*sizeof(uchar*));
    cudaMalloc(&hg_e, rows*sizeof(uchar*));
    cudaMalloc(&hb_e, rows*sizeof(uchar*));
    for(int i=0;i<rows;i++)
    {
        cudaMalloc(&hr_e_host[i], (gridDimx)*sizeof(uchar));
        cudaMalloc(&hg_e_host[i], (gridDimx)*sizeof(uchar));
        cudaMalloc(&hb_e_host[i], (gridDimx)*sizeof(uchar));
        uchar *temp_r, *temp_g, *temp_b;
        temp_r = (uchar *)malloc((gridDimx)*sizeof(uchar));
        temp_g = (uchar *)malloc((gridDimx)*sizeof(uchar));
        temp_b = (uchar *)malloc((gridDimx)*sizeof(uchar));
        for(int j=1;j<=gridDimx;j++)
        {
            int x = 16*j;
            if(x<columns)
            {
                temp_r[j-1] = r[i*columns+x];
                temp_g[j-1] = g[i*columns+x];
                temp_b[j-1] = b[i*columns+x];
            }
        }
        cudaMemcpy(hr_e_host[i], temp_r, gridDimx*sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(hg_e_host[i], temp_g, gridDimx*sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(hb_e_host[i], temp_b, gridDimx*sizeof(uchar), cudaMemcpyHostToDevice);
        free(temp_r);
        free(temp_g);
        free(temp_b);
    }
    cudaMemcpy(hr_e, hr_e_host, rows * sizeof(uchar*), cudaMemcpyHostToDevice);
    cudaMemcpy(hg_e, hg_e_host, rows * sizeof(uchar*), cudaMemcpyHostToDevice);
    cudaMemcpy(hb_e, hb_e_host, rows * sizeof(uchar*), cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(r_e_even, &hr_e, sizeof(uchar **));
    cudaMemcpyToSymbol(g_e_even, &hg_e, sizeof(uchar **));
    cudaMemcpyToSymbol(b_e_even, &hb_e, sizeof(uchar **));

    // Free the host-side arrays
    free(hr_e_host);
    free(hg_e_host);
    free(hb_e_host);
    cudaError_t errBefore = cudaGetLastError();
    if (errBefore != cudaSuccess) 
    {
        std::cerr << "Error before kernel launch: " << cudaGetErrorString(errBefore) << std::endl;
    }
    applySimpleLinearBlurFilter<<<grid, block>>>(r, g, b);
    cudaError_t errSync = cudaDeviceSynchronize();  // Wait for the kernel to finish
    if (errSync != cudaSuccess) 
    {
        std::cerr << "Synchronization error: " << cudaGetErrorString(errSync) << std::endl;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
    
    const int rows = img.rows;
    const int columns = img.cols;
    size_t size = sizeof(uchar) * rows * columns;

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *r, *g, *b;
    cudaMallocManaged(&r, size);
    cudaMallocManaged(&g, size);
    cudaMallocManaged(&b, size);
    
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < columns; ++x)
        {
            Vec3b rgb = img.at<Vec3b>(y, x);
            b[y*columns+x] = rgb.val[0];
            g[y*columns+x]= rgb.val[1];
            r[y*columns+x] = rgb.val[2];
        }
    }

    return {rows, columns, r, g, b};
}

__host__ std::tuple<uchar *, uchar *, uchar *>applyBlurKernel(std::string inputImage)
{
    cout << "CPU applying kernel\n";
    Mat img = imread(inputImage, IMREAD_COLOR);
    const int rows = img.rows;
    const int columns = img.cols;

    uchar *r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *b = (uchar *)malloc(sizeof(uchar) * rows * columns);

    for(int y = 0; y < rows; ++y)
    {
        for(int x = 1; x < columns-1; ++x)
        {
            Vec3b rgb0 = img.at<Vec3b>(y, x-1);
            Vec3b rgb1 = img.at<Vec3b>(y, x);
            Vec3b rgb2 = img.at<Vec3b>(y, x+1);
            b[y*columns+x] = (rgb0[0] + rgb1[0] + rgb2[0])/3;
            g[y*columns+x] = (rgb0[1] + rgb1[1] + rgb2[1])/3;
            r[y*columns+x] = (rgb0[2] + rgb1[2] + rgb2[2])/3;
        }
    }

    return {r, g, b};
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);
    try 
    {
        auto[rows, columns, r, g, b] = readImageFromFile(inputImage);
        allocateDeviceMemory(rows, columns);
        executeKernel(r, g, b, rows, columns, threadsPerBlock);

        Mat colorImage(rows, columns, CV_8UC3);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < columns; ++x)
            {
                colorImage.at<Vec3b>(y,x) = Vec3b(b[y*columns+x], g[y*columns+x], r[y*columns+x]);
            }
        }

        imwrite(outputImage, colorImage, compression_params);

        auto[test_r, test_g, test_b] = applyBlurKernel(inputImage);
        
        float scaledMeanDifferencePercentage = compareColorImages(r, g, b, test_r, test_g, test_b, rows, columns) * 100;
        cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";

        cleanUpDevice();
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}