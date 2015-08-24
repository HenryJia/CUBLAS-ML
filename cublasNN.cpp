#include "cublasNN.hpp"

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

double cublasNN::writeCSV(vector<vector<float>> data, string fileName)
{
	auto start = chrono::steady_clock::now();

	ofstream out(fileName);

	for(int i = 0; i < (data.size() - 1); i++)
	{
		for(int j = 0; j < (data[i].size() - 1); j++)
			out << data[i][j] << ',';
		out << data[i][data[i].size() - 1] << endl;
	}
	out << data[data.size() - 1][data[data.size() - 1].size() - 1];

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;

	return chrono::duration <double> (elapsed).count();
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

float* cublasNN::classToBin(float* a, int m)
{
	float* yBin = (float*)malloc(m * layers[layerNum - 1] * sizeof(yBin));

	for(int i = 0; i < m; i++)
		for(int j = 0; j < layers[layerNum - 1]; j++)
			yBin[IDX2C(i, j, m)] = 0;
	for(int i = 0; i < m; i++)
		yBin[(int)IDX2C(i, a[i], m)] = 1;

	return yBin;
}

void cublasNN::setData(vector<vector<float>> xVec, vector<vector<float>> yVec, bool classify)
{
	m = xVec.size();
	n = xVec[0].size();
	if(xOld)
	{
		free(x);
		free(y);
	}
	x = vector2dToMat(xVec);
	if(classify == false)
		y = vector2dToMat(yVec);
	else
	{
		float *yRaw = vector2dToMat(yVec);
		y = classToBin(yRaw, m);
		free(yRaw);
	}
	xOld = true;
}

void cublasNN::setValidateData(vector<vector<float>> xVec, vector<vector<float>> yVec, bool classify)
{
	mValidate = xVec.size();
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
	if(xPredictOld)
		free(xPredict);
	xPredict = vector2dToMat(xVec);
	xPredictOld = true;
}

void cublasNN::normaliseData()
{
	Mean = mean(x, m, n);
	Stddev = stddev(x, Mean, m, n);

	normalise(x, m, n);
}

void cublasNN::normalise(float* data, int a, int b)
{
	for(int i = 0; i < a; i++)
	{
		for(int j = 0; j < b; j++)
		{
			data[IDX2C(i, j, a)] = data[IDX2C(i, j, a)] - Mean[j];
			if(Stddev[j] != 0)
				data[IDX2C(i, j, a)] = data[IDX2C(i, j, a)] / Stddev[j];
			else
				data[IDX2C(i, j, a)] = 0;
		}
	}
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
	thetaSize[layerNum - 2] = (l[layerNum - 2] + 1) * l[layerNum - 1];
	totalThetaSize += thetaSize[layerNum - 2];

	if(thetaBaseGPUOld)
		cudaFree(thetaBaseGPU);

	cudaStat = cudaMalloc((void**)&thetaBaseGPU, totalThetaSize * sizeof(float));
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMalloc Failed" << endl;
		return;
	}
	thetaBaseGPUOld = true;
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
	xGPU = copyToGPU(x, m, (n + 1));
	yGPU = copyToGPU(y, m, layers[layerNum - 1]);
	xGPUOld = true;
}

float* cublasNN::copyToGPU(float* data, int a, int b)
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

float* cublasNN::copyFromGPU(float* dataGPU, int a, int b)
{
	float* data = (float*)malloc(a * b * sizeof(*data));
	if(data == nullptr)
	{
		cout << "malloc Failed" << endl;
		return nullptr;
	}

	cudaStat = cudaMemcpy(data, dataGPU, a * b * sizeof(*data), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMemcpy Failed" << endl;
		return nullptr;
	}

	return data;
}

void cublasNN::allocVarGPU(int batchNum)
{
	int totaldeltaSize = 0, totalaSize = 0, totalzSize = 0, totalDeltaSize = 0, mTotal = 0;
	int batchSize = m / batchNum;
	int mBatchMax;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));

	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

	deltaPos = (int*)malloc((layerNum - 1) * sizeof(*deltaPos));
	deltaSize = (int*)malloc((layerNum - 1) * sizeof(*deltaSize));

	DeltaPos = (int*)malloc((layerNum - 1) * sizeof(*DeltaPos));
	DeltaSize = (int*)malloc((layerNum - 1) * sizeof(*DeltaSize));

	// Split Data into mini-batches for mini-batch & stochastic methods

	xPosBatch = (int*)malloc(batchNum * sizeof(int));
	yPosBatch = (int*)malloc(batchNum * sizeof(int));
	mBatch = (int*)malloc(batchNum * sizeof(int));

	for(int i = 0; i < (batchNum - 1); i++)
	{
		xPosBatch[i] = i * batchSize * (n + 1);
		yPosBatch[i] = i * batchSize * layers[layerNum - 1];
		mBatch[i] = batchSize;
		mTotal += batchSize;
	}

	mBatch[batchNum - 1] = m - mTotal;
	mBatchMax = mBatch[batchNum - 1] > batchSize ? mBatch[batchNum - 1] : batchSize;
	xPosBatch[batchNum - 1] = (batchNum - 1) * batchSize * (n + 1);
	yPosBatch[batchNum - 1] = (batchNum - 1) * batchSize * layers[layerNum - 1];

	if(!zPos || !deltaPos || !zSize || !deltaSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	zPos[0] = 0;
	deltaPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		zSize[i] = layers[i + 1] * mBatchMax;
		deltaSize[i] = zSize[i];
		totalzSize += zSize[i];
		zPos[i + 1] = totalzSize;
		deltaPos[i + 1] = zPos[i + 1];
	}
	zSize[layerNum - 2] = layers[layerNum - 1] * mBatchMax;
	deltaSize[layerNum - 2] = zSize[layerNum - 2];
	totalzSize += zSize[layerNum - 2];
	totaldeltaSize = totalzSize;

	if(!aPos || !aSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	aPos[0] = 0;
	for(int i = 0; i < (layerNum - 2); i++)
	{
		aSize[i] = (layers[i + 1] + 1) * mBatchMax;
		totalaSize += aSize[i];
		aPos[i + 1] = totalaSize;
	}
	aSize[layerNum - 2] = layers[layerNum - 1] * mBatchMax;
	totalaSize += aSize[layerNum - 2];

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

	int zSizeMax = 0;
	for(int i = 0; i < layerNum - 1; i++)
		(zSizeMax < zSize[i] && (zSizeMax = zSize[i]));

	cudaMalloc((void**)&product, (zSizeMax + mBatchMax) * sizeof(float));
	cudaMalloc((void**)&sigGrad, zSizeMax * sizeof(float));
	cudaMalloc((void**)&JAll, aSize[layerNum - 2] * sizeof(float));
	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));
	cudaMalloc((void**)&deltaBaseGPU, totaldeltaSize * sizeof(float));
	cudaMalloc((void**)&DeltaBaseGPU, totalDeltaSize * sizeof(float));
}

void cublasNN::splitData(int batchNum)
{
	/* As xGPU and yGPU are stored in clumn major format, we cannot simply split into batches by moving the pointer,
	 * instead we will transpose the whole matrix, and then transpose each batch back. */

	cudaMalloc((void**)&xTransGPU, m * (n + 1) * sizeof(float));
	cudaMalloc((void**)&xSplitGPU, m * (n + 1) * sizeof(float));
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, n + 1, m, &alpha2, xGPU, m, &beta2, xGPU, m, xTransGPU, n + 1);
	for(int b = 0; b < batchNum; b++)
	{
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mBatch[b], n + 1, &alpha2, xTransGPU + xPosBatch[b],
		            n + 1, &beta2, xTransGPU + xPosBatch[b], n + 1, xSplitGPU + xPosBatch[b], mBatch[b]);
	}

	cudaMalloc((void**)&yTransGPU, m * layers[layerNum - 1] * sizeof(float));
	cudaMalloc((void**)&ySplitGPU, m * layers[layerNum - 1] * sizeof(float));
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, layers[layerNum - 1], m, &alpha2, yGPU, m, &beta2, yGPU, m,
	            yTransGPU, layers[layerNum - 1]);
	for(int b = 0; b < batchNum; b++)
	{
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mBatch[b], layers[layerNum - 1], &alpha2, yTransGPU + yPosBatch[b],
		            layers[layerNum - 1], &beta2, yTransGPU + yPosBatch[b], layers[layerNum - 1], ySplitGPU + yPosBatch[b], mBatch[b]);
	}

	cudaFree(xTransGPU);
	cudaFree(yTransGPU);
}

void cublasNN::releaseGPUVar()
{
	cudaFree(sigGrad);
	cudaFree(product);
	cudaFree(JAll);

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
}