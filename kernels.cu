#include "kernels.h"

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		B[i] = A[i] + alpha;
}

__global__ void kernelAbsVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		B[i] = abs(A[i]);
}

void scaVecAddGPU(const float* A, const float alpha, float* B, int M)
{
	kernelScaVecAdd<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, alpha, B, M);
}

void absVecGPU(const float* A, float* B, int M)
{
	kernelAbsVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}