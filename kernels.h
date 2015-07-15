#include <cuda.h>

#define CC 53
//#define NUM_BLOCKS 32
#define BLOCK_THREADS 64

#ifndef IDX2C
#define IDX2C(i,j,ld) j * ld + i // i is column, j is row, ld is total number of columns
#endif

#define NUM_BLOCKS(M) (M + BLOCK_THREADS - 1) / BLOCK_THREADS

/*
 * Declarations
 */

void scaVecAddGPU(const float* A, const float alpha, float* B, int M);
void absVecGPU(const float* A, float* B, int M);

__global__ void kernelScaVecAdd(const float* A, const float alpha, float* B, int M);

__global__ void kernelAbsVec(const float* A, float* B, int M);
