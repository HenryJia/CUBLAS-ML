#ifndef RANDINITWEIGHTS_H
#define RANDINITWEIGHTS_H

#include "cudadefs.h"
#include "cuda.h"

void randInitWeights1GPU(float* theta, int in, int out, int M);
void randInitWeights2GPU(float* theta, int in, int out, int M);

__global__ void kernelRandInitWeights1(float* theta, int in, int out, int M);
__global__ void kernelRandInitWeights2(float* theta, int in, int out, int M);

#endif // ACTIVATIONS_H