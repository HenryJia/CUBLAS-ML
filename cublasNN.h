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
	void setData(vector<vector<float>> xVec, vector<vector<float>> yVec, bool classify);
	void setValidateData(vector<vector<float>> xVec, vector<vector<float>> yVec, bool classify);
	void setPredictData(vector<vector<float>> xVec);
	void setLayers(int* layers, int lNum);
	void setIters(int i) { iters = i; }
	void setDisplay(int i) { display = i; }
	void setLambda(int l) { lambda = l; }
	void normaliseData() { normalise(x, m, n); }
	void normaliseValidateData() { normalise(xValidate, mValidate, nValidate); }
	void normalisePredictData() { normalise(xPredict, mValidate, nValidate); }
	void addBiasData() { float* temp = addBias(x, m, n); free(x); x = temp; }
	void addBiasDataValidate() { float* temp = addBias(xValidate, mValidate, nValidate); free(xValidate); xValidate = temp; }
	void addBiasDataPredict() { float* temp = addBias(xPredict, mPredict, nPredict); free(xPredict); xPredict = temp; }
	void copyDataGPU();
	double trainFuncApproxGradDescent(float rate, int batchNum = 1);
	double trainFuncApproxMomentum(float momentum, float rate, int batchNum = 1);
	double trainClassifyGradDescent(float rate, int batchNum = 1);
	double trainClassifyMomentum(float momentum, float rate, int batchNum = 1);

private:
	void splitData(int batchNum);
	float calcFinalCost(bool classify);
	void releaseGPUVar();
	float gradFuncApprox(int b /*short for batchNum*/);
	float gradClassify(int b /*short for batchNum*/);

	float* vector2dToMat(vector<vector<float>> data);
	float* classToBin(float* a, int m);
	void normalise(float* data, int a, int b);
	float* copyGPU(float* data, int a, int b);
	float writeCSV(string fileName, float* data);

	// CPU Linear Algebra Functions
	float* mean(float* data, int a, int b);
	float* stddev(float* data, float* mean, int a, int b);

	// GPU Linear Algebra Functions
	float* addBias(float* data, int a, int b);
	void allocVarGPU(int batchNum);
	void matMatMultiplyGPU(const float *A, const float *B, float *C, const int a, const int b, const int c,
	                       cublasOperation_t transa, cublasOperation_t transb, int lda, int ldb, int ldc);

	float alpha;
	float lambda;
	float JValidate;
	int iters;
	int display;
	int layerNum;

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
	int nValidate;
	int nPredict;

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
