#include "kernels.h"

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = A[i] + alpha;
}

__global__ void kernelVecVecSubtract(const float* A, const float* B, float* C, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		C[i] = A[i] - B[i];
}

__global__ void kernelVecVecElementMultiply(const float* A, const float* B, float* C, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		C[i] = A[i] * B[i];
}

__global__ void kernelVecVecElementDivide(const float* A, const float* B, float* C, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		C[i] = A[i] / B[i];
}

__global__ void kernelAbsVec(const float* A, float* B, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = abs(A[i]);
}

__global__ void kernelSigmoidVec(const float* A, float* B, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = 1 / (1 + exp(-A[i]));
}

__global__ void kernelSigmoidGradVec(const float* A, float* B, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float temp = 1 / (1 + exp(-A[i]));
		B[i] = temp * (1 - temp);
	}
}

__global__ void kernelSigmoidGradFromResultVec(const float* A, float* B, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = A[i] * (1 - A[i]);
}

__global__ void kernelSigmoidGrad2Vec(const float* A, float* B, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float temp = 1 / (1 + exp(-A[i]));
		B[i] = temp * (1 - temp) * (1 - 2 * temp);
	}
}

__global__ void kernelOnesVec(float* A, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		A[i] = 1.0f;
}

__global__ void kernelProbToNum(const float* hProb, float* hNum, const int M, const int N)
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

__global__ void kernelNegLnMaxCost(const float* h, const float* y, float *J, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		J[i] = -y[i] * log(h[i]) - (1 - y[i]) * log(1 - h[i]);
}

__global__ void kernelcountError(const float* h, const float* y, float* errors, const int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		errors[i] = h[i] != y[i] ? 1 : 0;
}

void scaVecAddGPU(const float* A, const float alpha, float* B, const int M)
{
	kernelScaVecAdd<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, alpha, B, M);
}

void vecVecSubtractGPU(const float* A, const float* B, float* C, const int M)
{
	kernelVecVecSubtract<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, C, M);
}

void vecVecElementMultiplyGPU(const float* A, const float* B, float* C, const int M)
{
	kernelVecVecElementMultiply<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, C, M);
}

void vecVecElementDivideGPU(const float* A, const float* B, float* C, const int M)
{
	kernelVecVecElementDivide<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, C, M);
}

void absVecGPU(const float* A, float* B, int M)
{
	kernelAbsVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidVecGPU(const float* A, float* B, const int M)
{
	kernelSigmoidVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidGradVecGPU(const float* A, float* B, const int M)
{
	kernelSigmoidGradVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidGradFromResultVecGPU(const float* A, float* B, const int M)
{
	kernelSigmoidGradFromResultVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidGrad2VecGPU(const float* A, float* B, const int M)
{
	kernelSigmoidGrad2Vec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void onesVecGPU(float* A, const int M)
{
	kernelOnesVec<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, M);
}

void probToNumGPU(const float* hProb, float* hNum, const int M, const int N)
{
	kernelProbToNum<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(hProb, hNum, M, N);
}

void negLnMaxCostGPU(const float* h, const float* y, float *J, const int M)
{
	kernelNegLnMaxCost<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, J, M);
}

void countErrorGPU(const float* h, const float* y, float* errors, const int M)
{
	kernelcountError<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, errors, M);
}