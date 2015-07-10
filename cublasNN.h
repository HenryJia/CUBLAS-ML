#ifndef CUBLASNN_H
#define CUBLASNN_H

#include <vector>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // i is column, j is row, ld is total number of columns

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
	void setLayers(int* layers, size_t lNum);
	void setClassify(bool c) { classification = c; }
	void setAlpha(double a) { alpha = a; }
	void setIters(int i) { iters = i; }
	void setLambda(int l) { lambda = l; }
	void normaliseData() { normalise(x, m, n); }
	void normaliseValidateData() { normalise(xValidate, mValidate, nValidate); }
	void normalisePredictData() { normalise(xPredict, mValidate, nValidate); }
private:
	float* vector2dToMat(vector<vector<float>> data);
	void normalise(float* data, size_t a, size_t b);
	float* mean(float* data, size_t a, size_t b);
	float* stddev(float* data, float* mean, size_t a, size_t b);
	//float trainFuncApprox();
	//float trainConcurrentFuncApprox();

	//float* randInitialiseWeights(int in, int out);
	float* sigmoid(float* data);
	float* sigmoidGradient(float* data);
	float writeCSV(string fileName, float* data);

	float alpha;
	float lambda;
	float JValidate;
	int iters;
	size_t layerNum;
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
	bool xOld;
	bool xValidateOld;
	bool xPredictOld;
	bool xGPUOld;
	bool xValidateGPUOld;
	bool xPredictGPUOld;
	bool thetaBaseGPUOld;
	int* layers;
	vector<float> J;
	vector<float> JBatch;
	vector<thread> t;
	bool classification;

	cudaError_t cudaStat;    
	cublasStatus_t stat;
	cublasHandle_t handle;

	size_t m;
	size_t mValidate;
	size_t mPredict;

	size_t n;
	size_t nValidate;
	size_t nPredict;

	vector<float*> z;
	vector<float*> a;

	vector<float*> delta;
	vector<float*> Delta;
	vector<float*> thetaGrad;

	vector<float*> aFinal;
	vector<float*> deltaFinal;
	vector<float*> DeltaFinal;
	vector<float*> thetaGradFinal;

	// Functions for concurrency
	void grad(size_t threadNum, int rangeLower, int rangeUpper);
	void sumthetaGrad();
};

#endif // CUBLASNN_H
