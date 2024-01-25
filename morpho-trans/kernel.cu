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

int main() {
    int option;
    std::cout << "wybierz operacje. 1- dylacja, 2- erozja, 3- otwieranie, 4- zamykanie, 5- morpological gradient, 6- top hat\n";
    std::cin >> option;

    cv::Mat image = cv::imread("C:\\Users\\micha\\source\\repos\\morpho-trans\\cos.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Nie można wczytać obrazu." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(uchar);

    uchar* hostInput = image.data;
    uchar* hostOutput = new uchar[imageSize];

    uchar* deviceInput;
    uchar* deviceOutput;

    cudaMalloc((void**)&deviceInput, imageSize);
    cudaMalloc((void**)&deviceOutput, imageSize);

    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    if (option == 1) {
        dilationKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else if (option == 2) {
        erosionKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
     else if (option == 3) {
        openingKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else if (option == 4) {
        closingKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else if (option == 5) {
    morphologicalGradientKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else if (option == 6) {
        topHatKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else {
        std::cout << "zle wejscie";
        return 0;
    }

    cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    cv::imshow("original", image);

    cv::Mat editedImage(height, width, CV_8UC1, hostOutput);
    cv::imshow("edited", editedImage);

    cv::waitKey(0);

    delete[] hostOutput;

    return 0;
}
