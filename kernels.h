#include <cuda.h>

#define CC 53
//#define NUM_BLOCKS 32
#define BLOCK_THREADS 64
#define BLOCK_DIM 16

#ifndef IDX2C
#define IDX2C(i,j,ld) j * ld + i // i is column, j is row, ld is total number of columns
#endif

#define NUM_BLOCKS(M) (M + BLOCK_THREADS - 1) / BLOCK_THREADS

void scaVecAddGPU(const float* A, const float alpha, float* B, const int M);
void vecVecSubtractGPU(const float* A, const float* B, float* C, int M);
void vecVecElementMultiplyGPU(const float* A, const float* B, float* C, const int M);
void vecVecElementDivideGPU(const float* A, const float* B, float* C, const int M);
void absVecGPU(const float* A, float* B, const int M);
void sigmoidVecGPU(const float* A, float* B, const int M);
void sigmoidGradVecGPU(const float* A, float* B, const int M);
void sigmoidGrad2VecGPU(const float* A, float* B, const int M);
void onesVecGPU(float* A, const int M);
void probToNumGPU(const float* hProb, float* hNum, const int M, const int N);
void negLnMaxCostGPU(const float* h, const float* y, float *J, const int M);
void countErrorGPU(const float* h, const float* y, float* errors, const int M);

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, const int M);
__global__ void kernelVecVecSubtract(const float* A, const float* B, float* C, const int M);
__global__ void kernelVecVecElementMultiply(const float* A, const float* B, float* C, const int M);
__global__ void kernelVecVecElementDivide(const float* A, const float* B, float* C, const int M);
__global__ void kernelAbsVec(const float* A, float* B, const int M);
__global__ void kernelSigmoidVec(const float* A, float* B, const int M);
__global__ void kernelSigmoidGradVec(const float* A, float* B, const int M);
__global__ void kernelSigmoidGrad2Vec(const float* A, float* B, const int M);
__global__ void kernelOnesVec(float* A, int M);
__global__ void kernelProbToNum(const float* hProb, float* hNum, const int M, const int N);
__global__ void kernelNegLnMaxCost(const float* h, const float* y, float *J, const int M);
__global__ void kernelcountError(const float* h, const float* y, float* errors, const int M);