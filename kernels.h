#include <cuda.h>



void scaVecAdd(const float* A, const float alpha, float* B, int M);
void absVec(const float* A, float* B, int M);

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M);

__global__ void kernelAbsVec(const float* A, float* B, int M);
