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
        int maxValue = 0; // Zmiana na maksymalną wartość

        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int newX = x + i;
                int newY = y + j;

                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    int pixelValue = inputImage[newY * width + newX];
                    maxValue = (pixelValue > maxValue) ? pixelValue : maxValue; // Zmiana na maksymalną wartość
                }
            }
        }

        outputImage[y * width + x] = maxValue; // Zapisanie maksymalnej wartości jako wynik erozji
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



int main() {
    // Wczytaj obraz za pomocą OpenCV
    int option;
    std::cout << "wybierz operacje. 1- dylacja, 2- erozja\n";
    std::cin >> option;

    cv::Mat image = cv::imread("C:\\Users\\micha\\source\\repos\\morpho-trans\\cos.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Nie można wczytać obrazu." << std::endl;
        return -1;
    }

    // Przygotuj dane na hostingu i urządzeniu
    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(uchar);

    uchar* hostInput = image.data;
    uchar* hostOutput = new uchar[imageSize];

    uchar* deviceInput;
    uchar* deviceOutput;

    cudaMalloc((void**)&deviceInput, imageSize);
    cudaMalloc((void**)&deviceOutput, imageSize);

    // Skopiuj dane z hostingu do urządzenia
    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);

    // Określ rozmiar bloków i siatki
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Wywołaj odpowiedni kernel do inwersji kolorów
    if (option == 1) {
        dilationKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else if (option == 2) {
        erosionKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, width, height, 5);
    }
    else {
        std::cout << "zle wejscie";
        return 0;
    }

    // Skopiuj wyniki z urządzenia do hostingu
    cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    // Zwolnij pamięć na urządzeniu
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Wyświetl oryginalny i przetworzony obraz
    cv::imshow("original", image);

    cv::Mat editedImage(height, width, CV_8UC1, hostOutput);
    cv::imshow("edited", editedImage);

    cv::waitKey(0);

    // Zwolnij pamięć na hoście
    delete[] hostOutput;

    return 0;
}
