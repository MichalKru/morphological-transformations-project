#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>


__global__ void dilationKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int halfKernel = kernelSize / 2;
        int maxValue = 0; 

        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = inputImage[newY * width + newX];
                    maxValue = (pixelValue > maxValue) ? pixelValue : maxValue; 
                }
            }
        }

        outputImage[y * width + x] = maxValue; 
    }
}

__global__ void erosionKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int halfKernel = kernelSize / 2;
        int minValue = 255;

        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = inputImage[newY * width + newX];
                    minValue = (pixelValue < minValue) ? pixelValue : minValue;
                }
            }
        }

        outputImage[y * width + x] = minValue;
    }
}

__global__ void openingKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int halfKernel = kernelSize / 2;
        int minValue = 255;

        // Erozja
        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = inputImage[newY * width + newX];
                    minValue = (pixelValue < minValue) ? pixelValue : minValue;
                }
            }
        }

        unsigned char erodedImage = minValue;

        // Dylatacja na wyniku erozji
        int maxValue = 0;

        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = erodedImage; 
                    maxValue = (pixelValue > maxValue) ? pixelValue : maxValue;
                }
            }
        }

        outputImage[y * width + x] = maxValue; 
    }
}

__global__ void closingKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int halfKernel = kernelSize / 2;
        int maxValue = 0;

        // Dylatacja
        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = inputImage[newY * width + newX];
                    maxValue = (pixelValue > maxValue) ? pixelValue : maxValue;
                }
            }
        }

        unsigned char dilatedImage = maxValue;

        // Erozja na wyniku dylatacji
        int minValue = 255;

        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = dilatedImage;
                    minValue = (pixelValue < minValue) ? pixelValue : minValue;
                }
            }
        }

        outputImage[y * width + x] = minValue;
    }
}

__global__ void morphologicalGradientKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int halfKernel = kernelSize / 2;
        int maxValue = 0;
        int minValue = 255;

        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = inputImage[newY * width + newX];
                    maxValue = (pixelValue > maxValue) ? pixelValue : maxValue;
                    minValue = (pixelValue < minValue) ? pixelValue : minValue;
                }
            }
        }

        outputImage[y * width + x] = maxValue - minValue;
    }
}

__global__ void topHatKernel(const unsigned char* inputImageOriginal, const unsigned char* inputImageOpening, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int originalValue = inputImageOriginal[y * width + x];
        int openinglValue = inputImageOpening[y * width + x];

        outputImage[y * width + x] = originalValue - openinglValue;
    }
}

__global__ void blackHatKernel(const unsigned char* inputImageOriginal, const unsigned char* inputImageClosing, unsigned char* outputImage, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int originalValue = inputImageOriginal[y * width + x];
        int closinglValue = inputImageClosing[y * width + x];

        outputImage[y * width + x] = originalValue - closinglValue;
    }
}

__global__ void invertColorsKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIdx = y * width + x;
        outputImage[pixelIdx] = 255 - inputImage[pixelIdx];
    }
}

int main() {
    int option;
    std::cout << "wybierz operacje. 1- dylatacja, 2- erozja, 3- zamykanie, 4- otwieranie, 5- morpological gradient, 6- top hat, 7- black hat, 8- invert colours\n";
    std::cin >> option;

    cv::Mat image = cv::imread("C:\\Users\\falek\\Documents\\balwan.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Nie można wczytać obrazu." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(uchar);

    uchar* hostInput = image.data;
    uchar* hostOutput = new uchar[imageSize];
    uchar* hostOutput2 = new uchar[imageSize];
    uchar* hostOutput3 = new uchar[imageSize];

    uchar* deviceInput;
    uchar* deviceOutput;
    uchar* deviceOutput2;
    uchar* deviceOutput3;

    cudaMalloc((void**)&deviceInput, imageSize);
    cudaMalloc((void**)&deviceOutput, imageSize);
    cudaMalloc((void**)&deviceOutput2, imageSize);
    cudaMalloc((void**)&deviceOutput3, imageSize);

    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    if (option == 1) {
        dilationKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);

        cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput);

    }
    else if (option == 2) {
        erosionKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);

        cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput);

    }
    else if (option == 3) {

        dilationKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 10);
        erosionKernel << <gridSize, blockSize >> > (deviceOutput, deviceOutput2, width, height, 10);

        cudaMemcpy(hostOutput, deviceOutput2, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput2);

    }
    else if (option == 4) {
        erosionKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 10);
        dilationKernel << <gridSize, blockSize >> > (deviceOutput, deviceOutput2, width, height, 10);

        cudaMemcpy(hostOutput, deviceOutput2, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput2);

    }
    else if (option == 5) {
        morphologicalGradientKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);

        cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput);

    }
    else if (option == 6) {
        erosionKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
        dilationKernel << <gridSize, blockSize >> > (deviceOutput, deviceOutput2, width, height, 5);
        topHatKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput2, deviceOutput3, width, height, 5);

        cudaMemcpy(hostOutput, deviceOutput3, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput3);

    }
    else if (option == 7) {
        dilationKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
        erosionKernel << <gridSize, blockSize >> > (deviceOutput, deviceOutput2, width, height, 5);
        blackHatKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput2, deviceOutput3, width, height, 5);

        cudaMemcpy(hostOutput, deviceOutput3, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput3);

    }
    else if (option == 8) {
        invertColorsKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height);

        cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
    }
    else {
        std::cout << "zle wejscie";
        return 0;
    }

    cv::imshow("original", image);

    cv::Mat editedImage(height, width, CV_8UC1, hostOutput);
    cv::imshow("edited", editedImage);

    cv::waitKey(0);

    delete[] hostOutput;

    return 0;
}
