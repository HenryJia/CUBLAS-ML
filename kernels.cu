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

__global__ void kernelSigmoid(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = 1 / (1 + exp(-A[i]));
}

__global__ void kernelSigmoidGrad(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float a = 1 / (1 + exp(-A[i]));
		B[i] = a * (1 - a);
	}
}

__global__ void kernelSoftmax(const float* A, float* B, int M, int N)
{
	// The idiot-proof un-optimised algorithm
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float total = 0;
		//extern __shared__ float e[];
		/*for(int j = 0; j < N; j++)
		{
			e[IDX2C(i, j, BLOCK_THREADS)] = exp(A[IDX2C(i, j, M)]);
			total += e[IDX2C(i, j, BLOCK_THREADS)];
		}
		for(int j = 0; j < N; j++)
			B[IDX2C(i, j, M)] = e[IDX2C(i, j, BLOCK_THREADS)] / total;*/
		for(int j = 0; j < N; j++)
			total += exp(A[IDX2C(i, j, M)]);
		for(int j = 0; j < N; j++)
			B[IDX2C(i, j, M)] = exp(A[IDX2C(i, j, M)]) / total;
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

__global__ void kernelCrossEntropyCost(float* h, float* y, float *J, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		J[i] = -y[i] * log(h[i]);
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

void sigmoidGPU(const float* A, float* B, int M)
{
	kernelSigmoid<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidGradGPU(const float* A, float* B, int M)
{
	kernelSigmoidGrad<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void softmaxGPU(const float* A, float* B, int M, int N)
{
	kernelSoftmax<<<NUM_BLOCKS(M), BLOCK_THREADS, BLOCK_THREADS * N * sizeof(float)>>>(A, B, M, N);
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

void crossEntropyCostGPU(float* h, float* y, float *J, int M)
{
	kernelCrossEntropyCost<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, J, M);
}

void countErrorGPU(float* h, float* y, float* errors, int M)
{
	kernelcountError<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, errors, M);
}