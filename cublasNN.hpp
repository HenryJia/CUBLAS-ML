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

#include "kernels.h"

#ifndef IDX2CS
#define IDX2C(i,j,ld) (j * ld + i) // i is column, j is row, ld is total number of columns
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
	void setLayers(int* layers, int lNum);
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

	template <typename func1>
	void randInitWeights(func1 formula);

	template <typename func1>
	double trainFuncApproxMomentum(float momentum, float rate, func1 activationHidden, func1 activationDerivative, int batchNum = 1);
	template <typename func1, typename func2, typename func3>
	double trainClassifyMomentum(float momentum, float rate, func1 activationHidden, func1 activationDerivative,
	                             func2 activationOutput, func3 costFunction, int batchNum = 1);

	template <typename func1>
	void validateFuncApprox(func1 activationHidden)
	{
		validate(false, activationHidden, NULL, NULL);
	}
	template <typename func1, typename func2, typename func3>
	void validateClassify(func1 activationHidden, func2 activationOutput, func3 costFunction)
	{
		validate(true, activationHidden, activationOutput, costFunction);
	}
	template <typename func1>
	vector<vector<float>> predictFuncApprox(func1 activationHidden)
	{
		return predict(false, activationHidden, NULL);
	}
	template <typename func1, typename func2>
	vector<vector<float>> predictClassify(func1 activationHidden, func2 activationOutput)
	{
		return predict(true, activationHidden, activationOutput);
	}

private:
	void splitData(int batchNum);
	void releaseGPUVar();

	template <typename func1> // Does not activate the last layer. That can be done by the caller of this function.
	void forwardPropagate(float* X, func1 activationHidden, int size);
	template <typename func1>
	void backwardPropagate(float *output, func1 activationDerivative, int b  /* short for batchNum */);

	// Note, activationOutput & costFunction is for classification. Set to NULL for function approximation.
	template <typename func1, typename func2, typename func3>
	float calcFinalCost(bool classify, func1 activationHidden, func2 activationOutput = NULL, func3 costFunction = NULL);
	template <typename func1, typename func2, typename func3>
	void validate(bool classify, func1 activationHidden, func2 activationOutput, func3 costFunction);
	template <typename func1, typename func2>
	vector<vector<float>> predict(bool classify, func1 activationHidden, func2 activationOutput);

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

	void allocVarGPU(int batchNum);
	// Wrapper for cublasSgemm to make life easier C(m,n) = A(m,k) * B(k,n)
	void matMatMultiplyGPU(const float *A, const float *B, float *C, const int a, const int b, const int c,
	                       cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N, int lda = -1, int ldb = -1,
	                       int ldc = -1)
	{
		const float alpha = 1, beta = 0;
		//int lda = m, int ldb = k, int ldc = m
		(lda < 0 && (lda = a));
		(ldb < 0 && (ldb = c));
		(ldc < 0 && (ldc = a));
	
		// Do the actual multiplication
		cublasSgemm(handle, transa, transb, a, b, c, &alpha, A, lda, B, ldb, &beta, C, ldc);
	}

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

// Definitions for template member functions

template <typename func1>
void cublasNN::randInitWeights(func1 formula)
{
	curandGenerateUniform(gen, thetaBaseGPU, totalThetaSize);
	for(int i = 0; i < (layerNum - 1); i++)
	{
		int in = layers[i] + 1;
		int out = layers[i + 1];
		//float epsilon = (/*sqrt(6)*/ 1 / sqrt(in/* + out*/));
		//float epsilon2 = /*2 * */ epsilon;
		//float negEpsilon = -epsilon;
		//float* theta = thetaBaseGPU + thetaPos[i];
		//cublasSscal(handle, thetaSize[i], &epsilon2, theta, 1);
		//scaVecAddGPU(theta, negEpsilon, theta, thetaSize[i]);
		//absGPU(theta, theta, thetaSize[i]);
		formula((thetaBaseGPU + thetaPos[i]), in, out, thetaSize[i]);
		float* temp = (float*)malloc(thetaSize[i] * sizeof(float));
		cudaMemcpy(temp, (thetaBaseGPU + thetaPos[i]), thetaSize[i] * sizeof(float), cudaMemcpyDeviceToHost);
		for(int j = 0; j < 5; j++)
		{
			for(int k = 0; k < 5; k++)
				cout << temp[IDX2C(j, k, (layers[i] + 1))] << '\t';
			cout << endl;
		}
		cout << endl;
		free(temp);
	}
}

template <typename func1>
double cublasNN::trainFuncApproxMomentum(float momentum, float rate, func1 activationHidden, func1 activationDerivative,
                                         int batchNum /*= 1*/)
{
	auto start = chrono::steady_clock::now();

	float* velocity;

	cudaMalloc((void**)&velocity, totalThetaSize * sizeof(float));
	cudaMemset(velocity, 0, totalThetaSize * sizeof(float));

	allocVarGPU(batchNum);

	splitData(batchNum);

	for(int i = 0; i < iters; i++)
	{
		for(int b = 0; b < batchNum; b++)
		{
			float alpha = -rate / mBatch[b];
			float J = 0;

			forwardPropagate((xSplitGPU + xPosBatch[b]), activationHidden, mBatch[b]);
			backwardPropagate((zBaseGPU + zPos[layerNum - 2]), activationDerivative, b);

			if((i + 1) % display == 0 && b == 0)
			{
				cublasSdot(handle, deltaSize[layerNum - 2], (deltaBaseGPU + deltaPos[layerNum - 2]), 1,
				           (deltaBaseGPU + deltaPos[layerNum - 2]), 1, &J);
				J /= (2 * mBatch[b]);
				cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;
			}

			for(int j = 0; j < layerNum - 1; j++)
			{
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &momentum, (velocity + thetaPos[j]),
				            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (velocity + thetaPos[j]),
				            (layers[j] + 1));
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (layers[j] + 1), layers[j + 1], &alpha2, (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1), &alpha2, (velocity + thetaPos[j]), (layers[j] + 1), (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1));
			}
		}
	}

	cudaFree(velocity);

	releaseGPUVar();

	calcFinalCost(false, activationHidden);

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

template <typename func1, typename func2, typename func3>
double cublasNN::trainClassifyMomentum(float momentum, float rate, func1 activationHidden, func1 activationDerivative,
                                       func2 activationOutput, func3 costFunction, int batchNum /*= 1*/)
{
	auto start = chrono::steady_clock::now();

	float* velocity;

	cudaMalloc((void**)&velocity, totalThetaSize * sizeof(float));
	cudaMemset(velocity, 0, totalThetaSize * sizeof(float));

	allocVarGPU(batchNum);

	splitData(batchNum);

	for(int i = 0; i < iters; i++)
	{
		for(int b = 0; b < batchNum; b++)
		{
			float alpha = -rate / mBatch[b];
			float J = 0;

			forwardPropagate((xSplitGPU + xPosBatch[b]), activationHidden, mBatch[b]);
			//sigmoidGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);
			//softmaxGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), mBatch[b], layers[layerNum - 1]);
			activationOutput((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), mBatch[b], layers[layerNum - 1]);
			backwardPropagate((aBaseGPU + aPos[layerNum - 2]), activationDerivative, b);
			/*float* temp = (float*)malloc(zSize[0] * sizeof(float));
			cudaMemcpy(temp, (zBaseGPU + zPos[0]), zSize[0] * sizeof(float), cudaMemcpyDeviceToHost);
			for(int j = 0; j < 5; j++)
			{
				for(int k = 0; k < 5; k++)
					cout << temp[IDX2C(j, k, mBatch[b])] << '\t';
				cout << endl;
			}
			free(temp);*/

			// Calculate cost
			if((i + 1) % display == 0 && b == 0)
			{
				//negLnMaxCostGPU((aBaseGPU + aPos[layerNum - 2]), ySplitGPU + yPosBatch[b], JAll, aSize[layerNum - 2]);
				//crossEntropyCostGPU((aBaseGPU + aPos[layerNum - 2]), (ySplitGPU + yPosBatch[b]), JAll, aSize[layerNum - 2]);
				costFunction((aBaseGPU + aPos[layerNum - 2]), (ySplitGPU + yPosBatch[b]), JAll, aSize[layerNum - 2]);
				cublasSasum(handle, aSize[layerNum - 2], JAll, 1, &J);
				J /= mBatch[b];
				cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;
			}

			for(int j = 0; j < layerNum - 1; j++)
			{
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &momentum, (velocity + thetaPos[j]),
				            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (velocity + thetaPos[j]),
				            (layers[j] + 1));
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (layers[j] + 1), layers[j + 1], &alpha2, (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1), &alpha2, (velocity + thetaPos[j]), (layers[j] + 1), (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1));
			}
		}
	}

	//cudaFree(velocity);

	releaseGPUVar();

	calcFinalCost(true, activationHidden, activationOutput, costFunction);

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

template <typename func1>
inline void cublasNN::forwardPropagate(float* X, func1 activationHidden, int size)
{
	// Forward Propagate
	matMatMultiplyGPU(X, (thetaBaseGPU + thetaPos[0]), (zBaseGPU + zPos[0]), size, layers[1], (layers[0] + 1));
	//sigmoidGPU((zBaseGPU + zPos[0]), (aBaseGPU + aPos[0] + size), zSize[0]);
	activationHidden((zBaseGPU + zPos[0]), (aBaseGPU + aPos[0] + size), zSize[0]);

	for(int j = 1; j < layerNum - 2; j++)
	{
		addBiasMatGPU((aBaseGPU + aPos[j - 1]), size);
		matMatMultiplyGPU((aBaseGPU + aPos[j - 1]), (thetaBaseGPU + thetaPos[j]), (zBaseGPU + zPos[j]), size, layers[j + 1],
		                  (layers[j] + 1));
		//sigmoidGPU((zBaseGPU + zPos[j]), (aBaseGPU + aPos[j] + size), zSize[j]);
		activationHidden((zBaseGPU + zPos[j]), (aBaseGPU + aPos[j] + size), zSize[j]);
	}
	addBiasMatGPU((aBaseGPU + aPos[layerNum - 3]), size);
	matMatMultiplyGPU((aBaseGPU + aPos[layerNum - 3]), (thetaBaseGPU + thetaPos[layerNum - 2]), (zBaseGPU + zPos[layerNum - 2]),
	                  size, layers[layerNum - 1], (layers[layerNum - 2] + 1));
	//sigmoidGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);
}

template <typename func1>
inline void cublasNN::backwardPropagate(float *output, func1 activationDerivative, int b)
{
	// Calculate the last delta
	vecVecSubtractGPU(output, ySplitGPU + yPosBatch[b], (deltaBaseGPU + deltaPos[layerNum - 2]), aSize[layerNum - 2]);

	// Calculate remaining deltas via backpropagation
	for(int j = layerNum - 3; j >= 0; j--)
	{
		//sigmoidGradGPU((zBaseGPU + zPos[j]), sigGrad, zSize[j]);
		activationDerivative((zBaseGPU + zPos[j]), sigGrad, zSize[j]);
		matMatMultiplyGPU((deltaBaseGPU + deltaPos[j + 1]), (thetaBaseGPU + thetaPos[j + 1]), product, mBatch[b],(layers[j + 1] + 1),
		                  layers[j + 2], CUBLAS_OP_N, CUBLAS_OP_T, mBatch[b], (layers[j + 1] + 1), mBatch[b]);
		vecVecElementMultiplyGPU((product + mBatch[b]), sigGrad, (deltaBaseGPU + deltaPos[j]), deltaSize[j]);
	}

	// Accumulate deltas to calculate Deltas
	matMatMultiplyGPU((deltaBaseGPU + deltaPos[0]), xSplitGPU + xPosBatch[b], (DeltaBaseGPU + DeltaPos[0]), (layers[1]), (layers[0] + 1),
	                  mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], (layers[1]));
	for(int j = 1; j < layerNum - 1; j++)
	matMatMultiplyGPU((deltaBaseGPU + deltaPos[j]), (aBaseGPU + aPos[j - 1]), (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]),
	                  (layers[j] + 1), mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], (layers[j + 1]));
}

template <typename func1, typename func2, typename func3>
float cublasNN::calcFinalCost(bool classify, func1 activationHidden, func2 activationOutput, func3 costFunction)
{
	float J = 0;
	int totalzSize = 0;
	int totalaSize = 0;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));


	zPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		zSize[i] = layers[i + 1] * m;
		totalzSize += zSize[i];
		zPos[i + 1] = totalzSize;
	}
	zSize[layerNum - 2] = layers[layerNum - 1] * m;
	totalzSize += zSize[layerNum - 2];

	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

	aPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		aSize[i] = (layers[i + 1] + 1) * m;
		totalaSize += aSize[i];
		aPos[i + 1] = totalaSize;
	}
	aSize[layerNum - 2] = layers[layerNum - 1] * m;
	totalaSize += aSize[layerNum - 2];

	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));
	cudaMalloc((void**)&JAll, aSize[layerNum - 2] * sizeof(float));

	forwardPropagate(xGPU, activationHidden, m);

	// Calculate the last delta
	if(classify == true)
	{
		//sigmoidGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);
		//negLnMaxCostGPU((aBaseGPU + aPos[layerNum - 2]), yGPU, JAll, aSize[layerNum - 2]);
		//softmaxGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), m, layers[layerNum - 1]);
		//crossEntropyCostGPU((aBaseGPU + aPos[layerNum - 2]), yGPU, JAll, aSize[layerNum - 2]);
		activationOutput((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), m, layers[layerNum - 1]);
		costFunction((aBaseGPU + aPos[layerNum - 2]), yGPU, JAll, aSize[layerNum - 2]);
		cublasSasum(handle, aSize[layerNum - 2], JAll, 1, &J);
		J /= m;
	}
	else
	{
		vecVecSubtractGPU((aBaseGPU + aPos[layerNum - 2]), yGPU, JAll, aSize[layerNum - 2]);
		cublasSdot(handle, aSize[layerNum - 2], JAll, 1, JAll, 1, &J);
		J /= (2 * m);
		cudaFree(deltaBaseGPU);
	}

	cout << "Final Cost: " << J << endl;

	if(classify == true)
	{
		float* predictNum;
		float* yNum;
		float* errors;
		float errorCount;
		cudaMalloc((void**)&errors, m * sizeof(float));
		cudaMalloc((void**)&predictNum, m * sizeof(float));
		cudaMalloc((void**)&yNum, m * sizeof(float));
		probToNumGPU((aBaseGPU + aPos[layerNum - 2]), predictNum, m, layers[layerNum - 1]);
		probToNumGPU(yGPU, yNum, m, layers[layerNum - 1]);
		countErrorGPU(predictNum, yNum, errors, m);
		cublasSasum(handle, m, errors, 1, &errorCount);
		cout << "Total errors " << '\t' << errorCount << endl << "Error frequency" << '\t' << (errorCount / m) << endl;
		cudaFree(errors);
		cudaFree(predictNum);
		cudaFree(yNum);
	}

	cudaFree(zBaseGPU);
	free(zPos);
	free(zSize);

	cudaFree(aBaseGPU);
	free(aPos);
	free(aSize);

	cudaFree(JAll);

	return J;
}

template <typename func1, typename func2, typename func3>
void cublasNN::validate(bool classify, func1 activationHidden, func2 activationOutput, func3 costFunction)
{
	float* yValGPU;
	float* yValBinGPU;
	float* xValGPU = copyToGPU(xValidate, mValidate, (n + 1));
	if(classify == true)
	{
		yValGPU = copyToGPU(yValidate, mValidate, 1);
		float* yValBin = classToBin(yValidate, mValidate);
		yValBinGPU = copyToGPU(yValBin, mValidate, layers[layerNum - 1]);
		free(yValBin);
	}
	else
		yValGPU = copyToGPU(yValidate, mValidate, layers[layerNum - 1]);

	float J = 0;
	int totalzSize = 0;
	int totalaSize = 0;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));


	zPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		zSize[i] = layers[i + 1] * mValidate;
		totalzSize += zSize[i];
		zPos[i + 1] = totalzSize;
	}
	zSize[layerNum - 2] = layers[layerNum - 1] * mValidate;
	totalzSize += zSize[layerNum - 2];

	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

	aPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		aSize[i] = (layers[i + 1] + 1) * mValidate;
		totalaSize += aSize[i];
		aPos[i + 1] = totalaSize;
	}
	aSize[layerNum - 2] = layers[layerNum - 1] * mValidate;
	totalaSize += aSize[layerNum - 2];

	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));
	cudaMalloc((void**)&JAll, aSize[layerNum - 2] * sizeof(float));

	forwardPropagate(xValGPU, activationHidden, mValidate);

	// Calculate the last delta
	if(classify == true)
	{
		//sigmoidGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);
		//negLnMaxCostGPU((aBaseGPU + aPos[layerNum - 2]), yValBinGPU, JAll, aSize[layerNum - 2]);
		//softmaxGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), mValidate, layers[layerNum - 1]);
		//crossEntropyCostGPU((aBaseGPU + aPos[layerNum - 2]), yValBinGPU, JAll, aSize[layerNum - 2]);
		activationOutput((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), mValidate, layers[layerNum - 1]);
		costFunction((aBaseGPU + aPos[layerNum - 2]), yValBinGPU, JAll, aSize[layerNum - 2]);
		cublasSasum(handle, aSize[layerNum - 2], JAll, 1, &J);
		J /= mValidate;
	}
	else
	{
		vecVecSubtractGPU((aBaseGPU + aPos[layerNum - 2]), yValGPU, JAll, aSize[layerNum - 2]);
		cublasSdot(handle, aSize[layerNum - 2], JAll, 1, JAll, 1, &J);
		J /= (2 * mValidate);
	}

	cout << "Final Validation Cost: " << J << endl;

	if(classify == true)
	{
		float* predictNum;
		float* errors;
		float errorCount;
		cudaMalloc((void**)&errors, mValidate * sizeof(float));
		cudaMalloc((void**)&predictNum, mValidate * sizeof(float));
		probToNumGPU((aBaseGPU + aPos[layerNum - 2]), predictNum, mValidate, layers[layerNum - 1]);
		countErrorGPU(predictNum, yValGPU, errors, mValidate);
		cublasSasum(handle, mValidate, errors, 1, &errorCount);
		cout << "Total validation errors " << '\t' << errorCount << endl;
		cout << "Validation error frequency" << '\t' << (errorCount / mValidate) << endl;
		cudaFree(errors);
		cudaFree(predictNum);
	}

	cudaFree(xValGPU);
	cudaFree(yValGPU);
	if(classify == true)
		cudaFree(yValBinGPU);

	cudaFree(zBaseGPU);
	free(zPos);
	free(zSize);

	cudaFree(aBaseGPU);
	free(aPos);
	free(aSize);

	cudaFree(JAll);
}

template <typename func1, typename func2>
vector<vector<float>> cublasNN::predict(bool classify, func1 activationHidden, func2 activationOutput)
{
	float* xPreGPU = copyToGPU(xPredict, mPredict, (n + 1));
	float* result;
	vector<vector<float>> resultVec;

	int totalzSize = 0;
	int totalaSize = 0;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));

	zPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		zSize[i] = layers[i + 1] * mPredict;
		totalzSize += zSize[i];
		zPos[i + 1] = totalzSize;
	}
	zSize[layerNum - 2] = layers[layerNum - 1] * mPredict;
	totalzSize += zSize[layerNum - 2];

	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

	aPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		aSize[i] = (layers[i + 1] + 1) * mPredict;
		totalaSize += aSize[i];
		aPos[i + 1] = totalaSize;
	}
	aSize[layerNum - 2] = layers[layerNum - 1] * mPredict;
	totalaSize += aSize[layerNum - 2];

	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));

	forwardPropagate(xPreGPU, activationHidden, mPredict);
	if(classify == true)
	{
		//sigmoidGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);
		//softmaxGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), mPredict, layers[layerNum - 1]);
		activationOutput((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), mPredict, layers[layerNum - 1]);
		float* predictNum;
		cudaMalloc((void**)&predictNum, mPredict * sizeof(float));
		probToNumGPU((aBaseGPU + aPos[layerNum - 2]), predictNum, mPredict, layers[layerNum - 1]);
		result = copyFromGPU(predictNum, mPredict, 1);
		cudaFree(predictNum);
	}
	else
		result = copyFromGPU((zBaseGPU + zPos[layerNum - 2]), mPredict, layers[layerNum - 1]);

	cudaFree(xPreGPU);

	cudaFree(zBaseGPU);
	free(zPos);
	free(zSize);

	cudaFree(aBaseGPU);
	free(aPos);
	free(aSize);

	const int outputs = classify == true ? 1 : layers[layerNum - 1];
	for(int i = 0; i < mPredict; i++)
	{
		vector<float> lineVec;
		for(int j = 0; j < outputs; j++)
			lineVec.push_back(result[IDX2C(i, j, mPredict)]);
		resultVec.push_back(lineVec);
	}
	return resultVec;
}