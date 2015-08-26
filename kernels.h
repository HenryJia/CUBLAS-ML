#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

#include "cudadefs.h"

void scaVecAddGPU(const float* A, const float alpha, float* B, int M);
void vecVecSubtractGPU(const float* A, float* B, float* C, int M);
void vecVecElementMultiplyGPU(const float* A, float* B, float* C, int M);
void absGPU(const float* A, float* B, int M);
void addBiasMatGPU(float* A, int M);
void probToNumGPU(float* hProb, float* hNum, int M, int N);
void countErrorGPU(float* h, float* y, float* errors, int M);

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M);
__global__ void kernelVecVecSubtract(const float* A, float* B, float* C, int M);
__global__ void kernelVecVecElementMultiply(const float* A, float* B, float* C, int M);
__global__ void kernelAbsVec(const float* A, float* B, int M);
__global__ void kernelAddBiasMat(float* A, int M);
__global__ void kernelProbToNum(float* hProb, float* hNum, int M, int N);
__global__ void kernelcountError(float* h, float* y, float* errors, int M);

#endif // KERNELS_H