#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <cuda.h>

#include "cudadefs.h"

void negLnMaxCostGPU(float* h, float* y, float *J, int M);
void crossEntropyCostGPU(float* h, float* y, float *J, int M);

__global__ void kernelNegLnMaxCost(float* h, float* y, float *J, int M);
__global__ void kernelCrossEntropyCost(float* h, float* y, float *J, int M);

#endif // COSTFUNCTIONS_H