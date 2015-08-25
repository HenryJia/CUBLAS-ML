#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cuda.h>

#include "cudadefs.h"

void sigmoidGPU(const float* A, float* B, int M);
void sigmoidOutputGPU(const float* A, float* B, int M, int N);
void sigmoidGradGPU(const float* A, float* B, int M);
void tanhGPU(const float* A, float* B, int M);
void sechSqGPU(const float* A, float* B, int M);
void softmaxGPU(const float* A, float* B, int M, int N);

__global__ void kernelSigmoid(const float* A, float* B, int M);
__global__ void kernelSigmoidGrad(const float* A, float* B, int M);
__global__ void kernelTanh(const float* A, float* B, int M);
__global__ void kernelSechSq(const float* A, float* B, int M);
__global__ void kernelSoftmax(const float* A, float* B, int M, int N);

#endif // ACTIVATIONS_H