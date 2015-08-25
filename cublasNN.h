#ifndef CUBLASNN_H
#define CUBLASNN_H

#include <vector>
#include <string>
#include <thread>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef IDX2CS
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // i is column, j is row, ld is total number of columns
#endif

using namespace std;

class cublasNN
{
public:
	cublasNN();
	~cublasNN();

	vector<vector<float>> readCSV(string fileName, bool header, float &time);
	double writeCSV(vector<vector<float>> data, string filename);
	void setData(vector<vector<float>> xVec, vector<vector<float>> yVec, bool classify);
	void setValidateData(vector<vector<float>> xVec, vector<vector<float>> yVec, bool classify);
	void setPredictData(vector<vector<float>> xVec);
	void setLayers(int* l, int lNum, void (*randInitWeights)(float* theta, int in, int out, int M));
	void setIters(int i) { iters = i; }
	void setDisplay(int i) { display = i; }
	void setLambda(int l) { lambda = l; }
	void normaliseData();
	void normaliseValidateData() { normalise(xValidate, mValidate, n); }
	void normalisePredictData() { normalise(xPredict, mValidate, n); }
	void addBiasData() { float* temp = addBias(x, m, n); free(x); x = temp; }
	void addBiasDataValidate() { float* temp = addBias(xValidate, mValidate, n); free(xValidate); xValidate = temp; }
	void addBiasDataPredict() { float* temp = addBias(xPredict, mPredict, n); free(xPredict); xPredict = temp; }
	void copyDataGPU();
	double trainFuncApproxMomentum(float momentum, float rate, void (*activationHidden)(const float*, float*, int),
	                               void (*activationDerivative)(const float*, float*, int), int batchNum = 1);
	double trainClassifyMomentum(float momentum, float rate, void (*activationHidden)(const float*, float*, int),
	                             void (*activationDerivative)(const float*, float*, int),
	                             void (*activationOutput)(const float*, float*, int, int),
	                             void (*costFunction)(float*, float*, float*, int), int batchNum = 1);
	void validateFuncApprox(void (*activationHidden)(const float*, float*, int))
	{
		validate(false, activationHidden, NULL, NULL);
	}
	void validateClassify(void (*activationHidden)(const float*, float*, int),
	                      void (*activationOutput)(const float*, float*, int, int),
	                      void (*costFunction)(float*, float*, float*, int))
	{
		validate(true, activationHidden, activationOutput, costFunction);
	}
	vector<vector<float>> predictFuncApprox()
	{
		return predict(false);
	}
	vector<vector<float>> predictClassify()
	{
		return predict(true);
	}

private:
	void splitData(int batchNum);
	float calcFinalCost(bool classify, void (*activationHidden)(const float*, float*, int),
	                    void (*activationOutput)(const float*, float*, int, int),
	                    void (*costFunction)(float*, float*, float*, int));
	void releaseGPUVar();
	void forwardPropagate(float* X, void (*activationHidden)(const float*, float*, int), int size); //Does not activate the last layer. That can be done by the caller of this function.
	void backwardPropagate(float *output, void (*activationDerivative)(const float*, float*, int), int b  /*short for batchNum*/);
	void validate(bool classify, void (*activationHidden)(const float*, float*, int),
	              void (*activationOutput)(const float*, float*, int, int),
	              void (*costFunction)(float*, float*, float*, int));
	vector<vector<float>> predict(bool classify);

	float* vector2dToMat(vector<vector<float>> data);
	float* classToBin(float* a, int m);
	void normalise(float* data, int a, int b);
	float* copyToGPU(float* data, int a, int b);
	float* copyFromGPU(float* dataGPU, int a, int b);
	float writeCSV(string fileName, float* data);

	// CPU Linear Algebra Functions
	float* addBias(float* data, int a, int b);
	float* mean(float* data, int a, int b);
	float* stddev(float* data, float* mean, int a, int b);

	// GPU Linear Algebra Functions
	void allocVarGPU(int batchNum);
	void matMatMultiplyGPU(const float *A, const float *B, float *C, const int a, const int b, const int c,
	                       cublasOperation_t transa, cublasOperation_t transb, int lda, int ldb, int ldc);

	float alpha;
	float lambda;
	float JValidate;
	int iters;
	int display;
	int layerNum;

	float* Mean;
	float* Stddev;

	float* x;
	float* xValidate;
	float* xPredict;
	float* y;
	float* yValidate;
	float* xGPU;
	float* xValidateGPU;
	float* xPredictGPU;
	float* yGPU;
	float* yValidateGPU;

	float* thetaBaseGPU;
	int* thetaPos;
	int* thetaSize;
	int totalThetaSize;


	bool xOld;
	bool xValidateOld;
	bool xPredictOld;
	bool xGPUOld;
	bool xValidateGPUOld;
	bool xPredictGPUOld;
	bool thetaBaseGPUOld;
	int* layers;

	cudaError_t cudaStat;    
	cublasStatus_t stat;
	cublasHandle_t handle;
	curandGenerator_t gen;

	const float alpha2 = 1.0f, beta2 = 0.0f;

	int m;
	int mValidate;
	int mPredict;

	int n; // Does not include the bias term

	// Will need to free the pointers below
	float* zBaseGPU;
	int* zPos;
	int* zSize;

	float* aBaseGPU;
	int* aPos;
	int* aSize;

	float* deltaBaseGPU;
	int* deltaPos;
	int* deltaSize;

	float* DeltaBaseGPU;
	int* DeltaPos;
	int* DeltaSize;

	float* product;
	float* sigGrad;
	float* JAll;

	// For mini-batch/stochastic gradient descent
	int* xPosBatch;
	float *xTransGPU ;
	float *xSplitGPU ;
	int* yPosBatch;
	float *yTransGPU;
	float *ySplitGPU;
	int* mBatch;
};

#endif // CUBLASNN_H
