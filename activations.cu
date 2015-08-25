#include "activations.h"


void sigmoidGPU(const float* A, float* B, int M)
{
	kernelSigmoid<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sigmoidOutputGPU(const float* A, float* B, int M, int N)
{
	kernelSigmoid<<<NUM_BLOCKS(M * N), BLOCK_THREADS>>>(A, B, M * N);
}

void sigmoidGradGPU(const float* A, float* B, int M)
{
	kernelSigmoidGrad<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void tanhGPU(const float* A, float* B, int M)
{
	kernelTanh<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void sechSqGPU(const float* A, float* B, int M)
{
	kernelSechSq<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(A, B, M);
}

void softmaxGPU(const float* A, float* B, int M, int N)
{
	kernelSoftmax<<<NUM_BLOCKS(M), BLOCK_THREADS, BLOCK_THREADS * N * sizeof(float)>>>(A, B, M, N);
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

__global__ void kernelTanh(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
		B[i] = tanh(A[i]);
}

__global__ void kernelSechSq(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float c = cosh(A[i]);
		B[i] = 1 / (c * c);
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