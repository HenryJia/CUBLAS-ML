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
		C[i] = B[i] - A[i];
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
		B[i] = 1 / (1 + exp(-A[i]));
}

__global__ void kernelSigmoidGradVec(const float* A, float* B, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
	{
		float temp = exp(-A[i]);
		B[i] = temp / (1 + temp) * (1 + temp);
	}
}

__global__ void kernelAddBiasMat(float* A, int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < M)
		A[i] = 1.0f;
}

__global__ void kernelTransposeMat(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
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

void transposeMatGPU(float *odata, float *idata, int size_x, int size_y)
{
	dim3 grid(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
	
	// warmup so we don't time CUDA startup
	kernelTransposeMat<<< grid, threads >>>(odata, idata, size_x, size_y);
}