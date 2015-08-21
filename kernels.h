#include <cuda.h>

#define CC 52
#define BLOCK_THREADS 64
#define BLOCK_DIM 16

#ifndef IDX2C
#define IDX2C(i,j,ld) (j * ld + i) // i is column, j is row, ld is total number of columns
#endif

#define NUM_BLOCKS(M) (M + BLOCK_THREADS - 1) / BLOCK_THREADS

void scaVecAddGPU(const float* A, const float alpha, float* B, int M);
void vecVecSubtractGPU(const float* A, float* B, float* C, int M);
void vecVecElementMultiplyGPU(const float* A, float* B, float* C, int M);
void absGPU(const float* A, float* B, int M);
void sigmoidGPU(const float* A, float* B, int M);
void sigmoidGradGPU(const float* A, float* B, int M);
void softmaxGPU(const float* A, float* B, int M, int N);
void addBiasMatGPU(float* A, int M);
void probToNumGPU(float* hProb, float* hNum, int M, int N);
void negLnMaxCostGPU(float* h, float* y, float *J, int M);
void crossEntropyCostGPU(float* h, float* y, float *J, int M);
void countErrorGPU(float* h, float* y, float* errors, int M);

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M);
__global__ void kernelVecVecSubtract(const float* A, float* B, float* C, int M);
__global__ void kernelVecVecElementMultiply(const float* A, float* B, float* C, int M);
__global__ void kernelAbsVec(const float* A, float* B, int M);
__global__ void kernelSigmoid(const float* A, float* B, int M);
__global__ void kernelSigmoidGrad(const float* A, float* B, int M);
__global__ void kernelSoftmax(const float* A, float* B, int M, int N);
__global__ void kernelAddBiasMat(float* A, int M);
__global__ void kernelProbToNum(float* hProb, float* hNum, int M, int N);
__global__ void kernelNegLnMaxCost(float* h, float* y, float *J, int M);
__global__ void kernelCrossEntropyCost(float* h, float* y, float *J, int M);
__global__ void kernelcountError(float* h, float* y, float* errors, int M);