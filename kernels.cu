#include "kernels.h"

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = A[i] + alpha;
}

__global__ void kernelVecVecSubtract(const float* A, float* B, float* C, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		C[i] = A[i] - B[i];
}

__global__ void kernelVecVecElementMultiply(const float* A, float* B, float* C, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		C[i] = B[i] * A[i];
}

__global__ void kernelAbsVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = abs(A[i]);
}

__global__ void kernelSigmoidVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = 1 / (1 + exp(-A[i]));
}

__global__ void kernelSigmoidGradVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float temp = 1 / (1 + exp(-A[i]));
		B[i] = temp * (1 - temp);
	}
}

__global__ void kernelAddBiasMat(float* A, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		A[i] = 1.0f;
}

__global__ void kernelProbToNum(float* hProb, float* hNum, int M, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float max = 0, maxNum = 0;
	if(i < M)
		for(int j = 0; j < N; j++)
			if(hProb[IDX2C(i, j, M)] > max)
			{
				max = hProb[IDX2C(i, j, M)];
				maxNum = j;
			}
	hNum[i] = maxNum;
}

__global__ void kernelNegLnMaxCost(float* h, float* y, float *J, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		J[i] = -y[i] * log(h[i]) - (1 - y[i]) * log(1 - h[i]);
}

__global__ void kernelcountError(float* h, float* y, float* errors, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		errors[i] = h[i] != y[i] ? 1.0f : 0.0f;
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

void probToNumGPU(float* hProb, float* hNum, int M, int N)
{
	kernelProbToNum<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(hProb, hNum, M, N);
}

void negLnMaxCostGPU(float* h, float* y, float *J, int M)
{
	kernelNegLnMaxCost<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, J, M);
}

void countErrorGPU(float* h, float* y, float* errors, int M)
{
	kernelcountError<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, errors, M);
}