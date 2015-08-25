#include "randinitweights.h"

void randInitWeights1GPU(float* theta, int in, int out, int M)
{
	kernelRandInitWeights1<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(theta, in, out, M);
}

void randInitWeights2GPU(float* theta, int in, int out, int M)
{
	kernelRandInitWeights2<<<NUM_BLOCKS(M), BLOCK_THREADS>>>(theta, in, out, M);
}

__global__ void kernelRandInitWeights1(float* theta, int in, int out, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float epsilon =  (1 / sqrt((float)in));
		theta[i] = theta[i] * 2 * epsilon - epsilon; // The means that the weights are initialised between [-epsilon, +ve epsilon]
	}
}

__global__ void kernelRandInitWeights2(float* theta, int in, int out, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < M)
	{
		float epsilon = sqrt(6.0f) / sqrt((float)(in + out));
		theta[i] = theta[i] * 2 * epsilon - epsilon; // The means that the weights are initialised between [-epsilon, +ve epsilon]
	}
}