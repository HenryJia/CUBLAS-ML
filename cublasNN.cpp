#include "cublasNN.h"
#include "kernels.h"

#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>

cublasNN::cublasNN()
{
	xOld = false;
	xValidateOld = false;
	xPredictOld = false;
	xGPUOld = false;
	xValidateGPUOld = false;
	xPredictGPUOld = false;
	thetaBaseGPUOld = false;
	iters = 100;
	alpha = 1;
	lambda = 0;

	cublasCreate(&handle);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long) clock());
}

cublasNN::~cublasNN()
{
	/*CPU Pointers to free
	* layers
	* x;
	* xValidate;
	* xPredict;
	* y;
	* yValidate;
	**/
	free(layers);
	free(x);
	free(xValidate);
	free(xPredict);
	free(y);
	free(yValidate);
	/*GPU Pointers to free
	* xGPU;
	* xValidateGPU;
	* xPredictGPU;
	* yGPU;
	* yValidateGPU;
	* thetaBaseGPU;
	* thetaPos
	* thetaSize
	**/
	cudaFree(xGPU);
	cudaFree(yGPU);
	cudaFree(thetaBaseGPU);
	cublasDestroy(handle);
}

vector<vector<float>> cublasNN::readCSV(string fileName, bool header, float &time)
{
	auto start = chrono::steady_clock::now();

	vector<vector<float>> result;
	ifstream in(fileName);
	string lineStr;
	string delimiter = ",";

	if(!in.is_open())
		cerr << "failed to open file\n";
	if(header == true)
		std::getline(in, lineStr);

	while(std::getline(in, lineStr))
	{
		vector<float> lineVec;
		size_t pos = 0;
		while((pos = lineStr.find(delimiter)) != std::string::npos)
		{
			lineVec.push_back(stold(lineStr.substr(0, pos)));
			lineStr.erase(0, pos + delimiter.length());
		}
		lineVec.push_back(stold(lineStr));
		result.push_back(lineVec);
	}

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	time = chrono::duration <float> (elapsed).count();

	return result;
}

float* cublasNN::vector2dToMat(vector<vector<float>> data)
{
	float* result;
	int a = data.size();
	int b = data[0].size();

	result = (float*)malloc(a * b * sizeof(*result));
	if(!result)
	{
		cout << "Malloc Failed" << endl;
		return nullptr;
	}
	for(int i = 0; i < a; i++)
		for(int j = 0; j < b; j++)
			result[IDX2C(i, j, a)] = data[i][j];

	return result;
}

void cublasNN::setData(vector<vector<float>> xVec, vector<vector<float>> yVec)
{
	m = xVec.size();
	n = xVec[0].size();
	if(xOld)
	{
		free(x);
		free(y);
	}
	x = vector2dToMat(xVec);
	y = vector2dToMat(yVec);
	xOld = true;
}

void cublasNN::setValidateData(vector<vector<float>> xVec, vector<vector<float>> yVec)
{
	mValidate = xVec.size();
	nValidate = xVec[0].size();
	if(xValidateOld)
	{
		free(xValidate);
		free(yValidate);
	}
	xValidate = vector2dToMat(xVec);
	yValidate = vector2dToMat(yVec);
	xValidateOld = true;
}

void cublasNN::setPredictData(vector<vector<float>> xVec)
{
	mPredict = xVec.size();
	nPredict = xVec[0].size();
	if(xPredictOld)
		free(xPredict);
	xPredict = vector2dToMat(xVec);
	xPredictOld = true;
}

void cublasNN::normalise(float* data, int a, int b)
{
	float* Mean;
	float* Stddev;
	Mean = mean(data, a, b);
	Stddev = stddev(data, Mean, a, b);

	for(int i = 0; i < a; i++)
	{
		for(int j = 0; j < b; j++)
		{
			data[IDX2C(i, j, a)] = data[IDX2C(i, j, a)] - Mean[j];
			data[IDX2C(i, j, a)] = data[IDX2C(i, j, a)] / Stddev[j];
		}
	}
	free(Mean);
	free(Stddev);
}

float* cublasNN::mean(float* data, int a, int b)
{
	float* result;

	result = (float*)malloc(b * sizeof(*result));
	if(!result)
	{
		cout << "Malloc Failed" << endl;
		return nullptr;
	}

	for(int i = 0; i < b; i++)
		result[i] = 0;

	for(int i = 0; i < a; i++)
		for(int j = 0; j < b; j++)
			result[j] += data[IDX2C(i, j, a)];

	for(int i = 0; i < b; i++)
		result[i] /= a;

	return result;
}

float* cublasNN::stddev(float* data, float* mean, int a, int b)
{
	float* result;

	result = (float*)malloc(b * sizeof(*result));
	if(!result)
	{
		cout << "Malloc Failed" << endl;
		return nullptr;
	}

	for(int i = 0; i < b; i++)
		result[i] = 0;

	for(int i = 0; i < a; i++)
	{
		for(int j = 0; j < b; j++)
		{
			float diff = data[IDX2C(i, j, a)] - mean[j];
			result[j] += diff * diff;
		}
	}

	for(int i = 0; i < b; i++)
	{
		result[i] /= (a - 1);
		result[i] = sqrt(result[i]);
	}

	return result;
}

void cublasNN::setLayers(int* l, int lNum)
{
	layerNum = lNum;
	layers = (int*)malloc(layerNum * sizeof(*layers));

	if(!layers)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	for(int i = 0; i < layerNum; i++)
		layers[i] = l[i];

	// To Randomly Initialise The Weights
	totalThetaSize = 0;
	thetaPos = (int*)malloc((layerNum - 1) * sizeof(*thetaPos));
	thetaSize = (int*)malloc((layerNum - 1) * sizeof(*thetaSize));
	if(!thetaPos || !thetaSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}
	thetaPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		thetaSize[i] = (l[i] + 1) * l[i + 1];
		totalThetaSize += thetaSize[i];
		thetaPos[i + 1] = totalThetaSize;
	}
	thetaSize[(layerNum - 2)] = (l[(layerNum - 2)] + 1) * l[(layerNum - 1)];
	totalThetaSize += thetaSize[(layerNum - 2)];

	if(thetaBaseGPUOld)
		cudaFree(thetaBaseGPU);

	cudaStat = cudaMalloc((void**)&thetaBaseGPU, totalThetaSize * sizeof(float));
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMalloc Failed" << endl;
		return;
	}
	thetaBaseGPUOld = true;

	curandGenerateUniform(gen, thetaBaseGPU, totalThetaSize);
	for(int i = 0; i < (layerNum - 1); i++)
	{
		int in = l[i] + 1;
		int out = l[i + 1];
		float epsilon = (sqrt(6) / sqrt(in + out));
		float epsilon2 = 2 * epsilon;
		float negEpsilon = -epsilon;
		float* theta = thetaBaseGPU + thetaPos[i];
		cublasSscal(handle, thetaSize[i], &epsilon2, theta, 1);
		scaVecAddGPU(theta, negEpsilon, theta, thetaSize[i]);
		absVecGPU(theta, theta, thetaSize[i]);
	}

	
}

float* cublasNN::addBias(float* data, int a, int b)
{
	float* result;
	result = (float*)malloc(a * (b + 1) * sizeof(*result));
	if(!result)
	{
		cout << "Malloc Failed" << endl;
		return nullptr;
	}
	for(int i = 0; i < a; i++)
		for(int j = 0; j < b; j++)
			result[IDX2C(i, j + 1, a)] = data[IDX2C(i, j, a)];

	for(int i = 0; i < a; i++)
		result[IDX2C(i, 0, a)] = 1.0f;
	return result;
}

void cublasNN::copyDataGPU()
{
	if(xGPUOld)
	{
		cudaFree(xGPU);
		cudaFree(yGPU);
	}
	xGPU = copyGPU(x, m, (n + 1));
	yGPU = copyGPU(y, m, 1);
	xGPUOld = true;
}

float* cublasNN::copyGPU(float* data, int a, int b)
{
	float* dataGPU;
	cudaStat = cudaMalloc((void**)&dataGPU, a * b * sizeof(*data));
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMalloc Failed" << endl;
		return nullptr;
	}

	cudaStat = cudaMemcpy(dataGPU, data, a * b * sizeof(*data), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMemcpy Failed" << endl;
		return nullptr;
	}
	return dataGPU;
}

void cublasNN::allocVarGPU(int batchNum)
{
	totalzSize = 0;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));

	totalaSize = 0;
	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

	totaldeltaSize = 0;
	deltaPos = (int*)malloc((layerNum - 1) * sizeof(*deltaPos));
	deltaSize = (int*)malloc((layerNum - 1) * sizeof(*deltaSize));

	totalDeltaSize = 0;
	DeltaPos = (int*)malloc((layerNum - 1) * sizeof(*DeltaPos));
	DeltaSize = (int*)malloc((layerNum - 1) * sizeof(*DeltaSize));

// Split Data into mini-batches for mini-batch & stochastic methods

	xPosBatch = (int*)malloc(batchNum * sizeof(int));
	mBatch = (int*)malloc(batchNum * sizeof(int));

	zPosBatch = (int**)malloc(batchNum * sizeof(int*));
	zSizeBatch = (int**)malloc(batchNum * sizeof(int*));

	aPosBatch = (int**)malloc(batchNum * sizeof(int*));
	aSizeBatch = (int**)malloc(batchNum * sizeof(int*));

	deltaPosBatch = (int**)malloc(batchNum * sizeof(int*));
	deltaSizeBatch = (int**)malloc(batchNum * sizeof(int*));


	for(int i = 0; i < batchNum; i++)
	{
		zPosBatch[i] = (int*)malloc((layerNum - 1) * sizeof(int));
		zSizeBatch[i] = (int*)malloc((layerNum - 1) * sizeof(int));

		aPosBatch[i] = (int*)malloc((layerNum - 1) * sizeof(int));
		aSizeBatch[i] = (int*)malloc((layerNum - 1) * sizeof(int));

		deltaPosBatch[i] = (int*)malloc((layerNum - 1) * sizeof(int));
		deltaSizeBatch[i] = (int*)malloc((layerNum - 1) * sizeof(int));
	}


	if(!zPos || !deltaPos || !zSize || !deltaSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	zPos[0] = 0;
	deltaPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		zSize[i] = layers[i + 1] * m;
		deltaSize[i] = zSize[i];
		totalzSize += zSize[i];
		zPos[i + 1] = totalzSize;
		deltaPos[i + 1] = zPos[i + 1];
	}
	zSize[(layerNum - 2)] = layers[(layerNum - 1)] * m;
	deltaSize[(layerNum - 2)] = zSize[(layerNum - 2)];
	totalzSize += zSize[(layerNum - 2)];
	totaldeltaSize = totalzSize;

	if(!aPos || !aSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	aPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		aSize[i] = (layers[i + 1] + 1) * m;
		totalaSize += aSize[i];
		aPos[i + 1] = totalaSize;
	}
	aSize[(layerNum - 2)] = layers[(layerNum - 1)] * m;
	totalaSize += aSize[(layerNum - 2)];

	if(!DeltaPos || !DeltaSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	DeltaPos[0] = 0;
	for(int i = 0; i < (layerNum - 1); i++)
	{
		DeltaSize[i] =  thetaSize[i];
		DeltaPos[i] =  thetaPos[i];
	}
	totalDeltaSize = totalThetaSize;

	int batchSize = m / batchNum;
	for(int i = 0; i < (batchNum - 1); i++)
	{
		xPosBatch[i] = i * batchSize;
		mBatch[i] = batchSize;
		for(int j = 0; j < (layerNum - 1); j++)
		{
			zPosBatch[i][j] = zPos[j] + (zSize[j] / m) * i * batchSize;
			zSizeBatch[i][j] = zSize[j] / m * batchSize;

			aPosBatch[i][j] = aPos[j] + (aSize[j] / m) * i * batchSize;
			aSizeBatch[i][j] = aSize[j] / m * batchSize;

			deltaPosBatch[i][j] = deltaPos[j] + (deltaSize[j] / m) * i * batchSize;
			deltaSizeBatch[i][j] = deltaSize[j] / m * batchSize;
		}
	}

	xPosBatch[batchNum - 1] = (batchNum - 1) * batchSize;
	mBatch[batchNum - 1] = m - xPosBatch[batchNum - 1];
	for(int j = 0; j < (layerNum - 1); j++)
	{
		zPosBatch[batchNum - 1][j] = zPos[j] + (zSize[j] / m) * (batchNum - 1) * batchSize;
		zSizeBatch[batchNum - 1][j] = m - zPosBatch[batchNum - 1][j];

		aPosBatch[batchNum - 1][j] = aPos[j] + (aSize[j] / m) * (batchNum - 1) * batchSize;
		aSizeBatch[batchNum - 1][j] = m - aPosBatch[batchNum - 1][j];

		deltaPosBatch[batchNum - 1][j] = deltaPos[j] + (deltaSize[j] / m) * (batchNum - 1) * batchSize;
		deltaSizeBatch[batchNum - 1][j] = m - deltaPosBatch[batchNum - 1][j];
	}


	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));
	cudaMalloc((void**)&deltaBaseGPU, totaldeltaSize * sizeof(float));
	cudaMalloc((void**)&DeltaBaseGPU, totalDeltaSize * sizeof(float));
}

// Wrapper for cublasSgemm to make life easier C(m,n) = A(m,k) * B(k,n)
void cublasNN::matMatMultiplyGPU(const float *A, const float *B, float *C, const int a, const int b, const int c,
									cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N,
									int lda = -1, int ldb = -1, int ldc = -1)
{
	const float alpha = 1, beta = 0;
	//int lda = m, int ldb = k, int ldc = m
	(lda < 0 && (lda = a));
	(ldb < 0 && (ldb = c));
	(ldc < 0 && (ldc = a));

	// Do the actual multiplication
	cublasSgemm(handle, transa, transb, a, b, c, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

double cublasNN::trainFuncApproxGradDescent(float rate, int batchNum /*= 1*/)
{
	auto start = chrono::steady_clock::now();
	allocVarGPU(batchNum);
	float J;
	float* product;
	float* sigGrad;
	const float alpha2 = 1.0f, alpha = -rate / m;

	int zSizeMax = 0;
	for(int i = 0; i < layerNum - 1; i++)
		(zSizeMax < zSize[i] && (zSizeMax = zSize[i]));
	cudaMalloc((void**)&product, (zSizeMax + m) * sizeof(float));
	cudaMalloc((void**)&sigGrad, zSizeMax * sizeof(float));

	for(int i = 0; i < iters; i++)
	{
		for(int b = 0; b < batchNum; b++)
		{
			// Forward Propagate
			matMatMultiplyGPU(xGPU + xPosBatch[b], (thetaBaseGPU + thetaPos[0]), (zBaseGPU + zPosBatch[b][0]), mBatch[b], layers[1], (layers[0] + 1));
			sigmoidVecGPU((zBaseGPU + zPosBatch[b][0]), (aBaseGPU + aPosBatch[b][0] + mBatch[b]), zSizeBatch[b][0]);
			for(int j = 1; j < layerNum - 2; j++)
			{
				addBiasMatGPU((aBaseGPU + aPosBatch[b][j - 1]), mBatch[b]);
				matMatMultiplyGPU((aBaseGPU + aPosBatch[b][j - 1]), (thetaBaseGPU + thetaPos[j]), (zBaseGPU + zPosBatch[b][j]),
				                  mBatch[b], layers[j + 1], (layers[j] + 1));
				sigmoidVecGPU((zBaseGPU + zPosBatch[b][j]), (aBaseGPU + aPosBatch[b][j] + mBatch[b]), zSizeBatch[b][j]);
			}
			addBiasMatGPU((aBaseGPU + aPosBatch[b][layerNum - 3]), mBatch[b]);
			matMatMultiplyGPU((aBaseGPU + aPosBatch[b][layerNum - 3]), (thetaBaseGPU + thetaPos[layerNum - 2]),
			                  (zBaseGPU + zPosBatch[b][layerNum - 2]), mBatch[b], layers[layerNum - 1], (layers[layerNum - 2] + 1));
			//sigmoidVecGPU((zBaseGPU + zPosBatch[b][layerNum - 2]), (aBaseGPU + aPosBatch[b][layerNum - 2]), zSizeBatch[b][layerNum - 2]);

			// Calculate the last delta
			vecVecSubtractGPU((zBaseGPU + zPosBatch[b][layerNum - 2]), yGPU, (deltaBaseGPU + deltaPosBatch[b][layerNum - 2]), zSizeBatch[b][layerNum - 2]);

			// Calculate cost
			cublasSdot(handle, deltaSizeBatch[b][layerNum - 2], (deltaBaseGPU + deltaPosBatch[b][layerNum - 2]), 1,
			           (deltaBaseGPU + deltaPosBatch[b][layerNum - 2]), 1, &J);
			J /= (2 * mBatch[b]);

			if((i + 1) % display == 0 && b == 0)
				cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;

			// Calculate remaining deltas via backpropagation
			for(int j = layerNum - 3; j >= 0; j--)
			{
				sigmoidGradVecGPU((zBaseGPU + zPosBatch[b][j]), sigGrad, zSizeBatch[b][j]);
				matMatMultiplyGPU((deltaBaseGPU + deltaPosBatch[b][j + 1]), (thetaBaseGPU + thetaPos[j + 1]), product,
				                  mBatch[b], (layers[j + 1] + 1), layers[j + 2], CUBLAS_OP_N, CUBLAS_OP_T, m, (layers[j + 1] + 1), mBatch[b]);
			
				vecVecElementMultiplyGPU((product + mBatch[b]), sigGrad, (deltaBaseGPU + deltaPosBatch[b][j]), deltaSizeBatch[b][j]);
			}

			// Accumulate deltas to calculate Deltas
			matMatMultiplyGPU((deltaBaseGPU + deltaPosBatch[b][0]), xGPU + xPosBatch[b], (DeltaBaseGPU + DeltaPos[0]),
			                  (layers[1]), (layers[0] + 1), mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], (layers[1]));
			for(int j = 1; j < layerNum - 1; j++)
				matMatMultiplyGPU((deltaBaseGPU + deltaPosBatch[b][j]), (aBaseGPU + aPosBatch[b][j - 1]), (DeltaBaseGPU + DeltaPos[j]),
				                  (layers[j + 1]), (layers[j] + 1), mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], (layers[j + 1]));

			// Calculate gradients from Deltas and perform gradient descent
			for(int j = 0; j < layerNum - 1; j++)
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &alpha2, (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1));
		}
	}

	cudaFree(sigGrad);
	cudaFree(product);

	cudaFree(zBaseGPU);
	free(zPos);
	free(zSize);

	cudaFree(aBaseGPU);
	free(aPos);
	free(aSize);

	cudaFree(deltaBaseGPU);
	free(deltaPos);
	free(deltaSize);

	cudaFree(DeltaBaseGPU);
	free(DeltaPos);
	free(DeltaSize);

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}