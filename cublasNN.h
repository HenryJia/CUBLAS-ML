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
	void setData(vector<vector<float>> xVec, vector<vector<float>> yVec);
	void setValidateData(vector<vector<float>> xVec, vector<vector<float>> yVec);
	void setPredictData(vector<vector<float>> xVec);
	void setLayers(int* layers, int lNum);
	void setClassify(bool c) { classification = c; }
	void setAlpha(double a) { alpha = a; }
	void setIters(int i) { iters = i; }
	void setLambda(int l) { lambda = l; }
	void normaliseData() { normalise(x, m, n); }
	void normaliseValidateData() { normalise(xValidate, mValidate, nValidate); }
	void normalisePredictData() { normalise(xPredict, mValidate, nValidate); }
	void addBiasData() { float* temp = addBias(x, m, n); free(x); x = temp; }
	void addBiasDataValidate() { float* temp = addBias(xValidate, mValidate, nValidate); free(xValidate); xValidate = temp; }
	void addBiasDataPredict() { float* temp = addBias(xPredict, mPredict, nPredict); free(xPredict); xPredict = temp; }
	void copyDataGPU();
	double trainFuncApprox();

private:
	float* vector2dToMat(vector<vector<float>> data);
	void normalise(float* data, int a, int b);
	float* copyGPU(float* data, int a, int b);
	float writeCSV(string fileName, float* data);

	// CPU Linear Algebra Functions
	float* mean(float* data, int a, int b);
	float* stddev(float* data, float* mean, int a, int b);
	//float trainFuncApprox();
	//float trainConcurrentFuncApprox();

	//float* randInitialiseWeights(int in, int out);
	// GPU Linear Algebra Functions
	float* addBias(float* data, int a, int b);
	void allocVarGPU();
	void matMatMultiplyGPU(const float *A, const float *B, float *C, const int a, const int b, const int c,
							cublasOperation_t transa, cublasOperation_t transb, int lda, int ldb, int ldc);

	float alpha;
	float lambda;
	float JValidate;
	int iters;
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
	//vector<float> J;
	//vector<float> JBatch;
	//vector<thread> t;
	bool classification;

	cudaError_t cudaStat;    
	cublasStatus_t stat;
	cublasHandle_t handle;
	curandGenerator_t gen;

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
	int totalzSize;

	float* aBaseGPU;
	int* aPos;
	int* aSize;
	int totalaSize;

	float* deltaBaseGPU;
	int* deltaPos;
	int* deltaSize;
	int totaldeltaSize;

	float* DeltaBaseGPU;
	int* DeltaPos;
	int* DeltaSize;
	int totalDeltaSize;

	/*vector<float*> aFinal;
	vector<float*> deltaFinal;
	vector<float*> DeltaFinal;
	vector<float*> thetaGradFinal;*/

	// Functions for concurrency
	void grad(size_t threadNum, int rangeLower, int rangeUpper);
	void sumthetaGrad();
};

#endif // CUBLASNN_H
