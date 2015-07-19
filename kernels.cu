#include "kernels.h"

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		B[i] = A[i] + alpha;
}

__global__ void kernelVecVecSubtract(const float* A, float* B, float* C, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		C[i] = A[i] - B[i];
}

__global__ void kernelVecVecElementMultiply(const float* A, float* B, float* C, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		C[i] = B[i] * A[i];
}

__global__ void kernelAbsVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		B[i] = abs(A[i]);
}

__global__ void kernelSigmoidVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		B[i] = 1 / (1 + expf(-A[i]));
}

__global__ void kernelSigmoidGradVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
	{
		float temp = 1 / (1 + expf(-A[i]));
		B[i] = temp * (1 - temp);
	}
}

__global__ void kernelAddBiasMat(float* A, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		A[i] = 1.0f;
}

void scaVecAddGPU(const float* A, const float alpha, float* B, int M)
{
	kernelScaVecAdd<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, alpha, B, M);
}

void vecVecSubtractGPU(const float* A, float* B, float* C, int M)
{
	kernelVecVecSubtract<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, C, M);
}

void vecVecElementMultiplyGPU(const float* A, float* B, float* C, int M)
{
	kernelVecVecElementMultiply<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, C, M);
}

void absVecGPU(const float* A, float* B, int M)
{
	kernelAbsVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidVecGPU(const float* A, float* B, int M)
{
	kernelSigmoidVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidGradVecGPU(const float* A, float* B, int M)
{
	kernelSigmoidGradVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void addBiasMatGPU(float* A, int M)
{
	kernelAddBiasMat<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, M);
}