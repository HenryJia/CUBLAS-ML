#include <cuda.h>

#define CC 53
//#define NUM_BLOCKS 32
#define BLOCK_THREADS 64
#define BLOCK_DIM 16

#ifndef IDX2C
#define IDX2C(i,j,ld) j * ld + i // i is column, j is row, ld is total number of columns
#endif

#define NUM_BLOCKS(M) (M + BLOCK_THREADS - 1) / BLOCK_THREADS

void scaVecAddGPU(const float* A, const float alpha, float* B, int M);
void vecVecSubtractGPU(const float* A, float* B, float* C, int M);
void vecVecElementMultiplyGPU(const float* A, float* B, float* C, int M);
void absVecGPU(const float* A, float* B, int M);
void sigmoidVecGPU(const float* A, float* B, int M);
void sigmoidGradVecGPU(const float* A, float* B, int M);
void onesVecGPU(float* A, int M);
void probToNumGPU(float* hProb, float* hNum, int M, int N);
void negLnMaxCostGPU(float* h, float* y, float *J, int M);
void countErrorGPU(float* h, float* y, float* errors, int M);

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M);
__global__ void kernelVecVecSubtract(const float* A, float* B, float* C, int M);
__global__ void kernelVecVecElementMultiply(const float* A, float* B, float* C, int M);
__global__ void kernelAbsVec(const float* A, float* B, int M);
__global__ void kernelSigmoidVec(const float* A, float* B, int M);
__global__ void kernelSigmoidGradVec(const float* A, float* B, int M);
__global__ void kernelOnesVec(float* A, int M);
__global__ void kernelProbToNum(float* hProb, float* hNum, int M, int N);
__global__ void kernelNegLnMaxCost(float* h, float* y, float *J, int M);
__global__ void kernelcountError(float* h, float* y, float* errors, int M);