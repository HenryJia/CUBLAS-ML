#include "costfunctions.h"


void negLnMaxCostGPU(float* h, float* y, float *J, int M)
{
	kernelNegLnMaxCost<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, J, M);
}

void crossEntropyCostGPU(float* h, float* y, float *J, int M)
{
	kernelCrossEntropyCost<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(h, y, J, M);
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