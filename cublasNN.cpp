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
			if(Stddev[j] != 0)
				data[IDX2C(i, j, a)] = data[IDX2C(i, j, a)] / Stddev[j];
			else
				data[IDX2C(i, j, a)] = 0;
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
	yGPU = copyGPU(y, m, layers[layerNum - 1]);
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

void cublasNN::allocVar(int batchNum, bool hessian)
{
	int totaldeltaSize = 0, totalaSize = 0, totalzSize = 0, totalDeltaSize = 0, totalsigGradSize = 0, totalproductSize = 0, mTotal = 0;
	int batchSize = m / batchNum;
	int mBatchMax, zSizeMax = 0;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));

	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

	deltaPos = (int*)malloc((layerNum - 1) * sizeof(*deltaPos));
	deltaSize = (int*)malloc((layerNum - 1) * sizeof(*deltaSize));

	DeltaPos = (int*)malloc((layerNum - 1) * sizeof(*DeltaPos));
	DeltaSize = (int*)malloc((layerNum - 1) * sizeof(*DeltaSize));

	sigGradPos = (int*)malloc((layerNum - 2) * sizeof(*sigGradPos));
	sigGradSize = (int*)malloc((layerNum - 2) * sizeof(*sigGradSize));

	productPos = (int*)malloc((layerNum - 2) * sizeof(*productPos));
	productSize = (int*)malloc((layerNum - 2) * sizeof(*productSize));

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
		sigGradSize[i] = zSize[i];
		totalzSize += zSize[i];
		zPos[i + 1] = totalzSize;
		deltaPos[i + 1] = zPos[i + 1];
		sigGradPos[i] = zPos[i];
	}
	totalsigGradSize = totalzSize;
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
		productSize[i] = aSize[i];
		totalaSize += aSize[i];
		aPos[i + 1] = totalaSize;
		productPos[i] = aPos[i];
	}
	totalproductSize = totalaSize;
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

	for(int i = 0; i < layerNum - 1; i++)
		(zSize[i] > zSizeMax && (zSizeMax = zSize[i]));

	cudaMalloc((void**)&productBaseGPU, totalproductSize * sizeof(float));
	cudaMalloc((void**)&sigGradBaseGPU, totalsigGradSize * sizeof(float));
	cudaMalloc((void**)&JAll, aSize[layerNum - 2] * sizeof(float));
	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));
	cudaMalloc((void**)&deltaBaseGPU, totaldeltaSize * sizeof(float));
	cudaMalloc((void**)&DeltaBaseGPU, totalDeltaSize * sizeof(float));

	if(hessian)
	{
		int thetaSizeMax = 0;

		delta2Pos = (int*)malloc((layerNum - 1) * sizeof(*deltaPos));
		delta2Size = (int*)malloc((layerNum - 1) * sizeof(*deltaSize));

		Delta2Pos = (int*)malloc((layerNum - 1) * sizeof(*DeltaPos));
		Delta2Size = (int*)malloc((layerNum - 1) * sizeof(*DeltaSize));

		for(int i = 0; i < (layerNum - 1); i++)
		{
			delta2Pos[i] = zPos[i];
			delta2Size[i] = zSize[i];

			Delta2Pos[i] = thetaPos[i];
			Delta2Size[i] = thetaSize[i];
		}

		for(int i = 0; i < (layerNum - 1); i++)
			(thetaSize[i] > thetaSizeMax && (thetaSizeMax = thetaSize[i]));

		cudaMalloc((void**)&productHess, (zSizeMax + mBatchMax) * sizeof(float));
		cudaMalloc((void**)&aSq, (zSizeMax + mBatchMax) * sizeof(float));
		cudaMalloc((void**)&sigGrad2, zSizeMax * sizeof(float));
		cudaMalloc((void**)&sigGradSq, zSizeMax * sizeof(float));
		cudaMalloc((void**)&thetaSq, thetaSizeMax * sizeof(float));
		cudaMalloc((void**)&delta2BaseGPU, totaldeltaSize * sizeof(float));
		cudaMalloc((void**)&Delta2BaseGPU, totalDeltaSize * sizeof(float));
	}
}

void cublasNN::splitData(int batchNum)
{
	/* As xGPU and yGPU are stored in clumn major format, we cannot simply split into batches by moving the pointer,
	 * instead we will transpose the whole matrix, and then transpose each batch back. */

	cudaMalloc((void**)&xTransGPU, m * (n + 1) * sizeof(float));
	cudaMalloc((void**)&xSplitGPU, m * (n + 1) * sizeof(float));
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, n + 1, m, &one, xGPU, m, &zero, xGPU, m, xTransGPU, n + 1);
	for(int b = 0; b < batchNum; b++)
	{
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mBatch[b], (n + 1), &one, (xTransGPU + xPosBatch[b]),
		            (n + 1), &zero, (xTransGPU + xPosBatch[b]), (n + 1), (xSplitGPU + xPosBatch[b]), mBatch[b]);
	}

	cudaMalloc((void**)&yTransGPU, m * layers[layerNum - 1] * sizeof(float));
	cudaMalloc((void**)&ySplitGPU, m * layers[layerNum - 1] * sizeof(float));
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, layers[layerNum - 1], m, &one, yGPU, m, &zero, yGPU, m,
	            yTransGPU, layers[layerNum - 1]);
	for(int b = 0; b < batchNum; b++)
	{
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mBatch[b], layers[layerNum - 1], &one, yTransGPU + yPosBatch[b],
		            layers[layerNum - 1], &zero, yTransGPU + yPosBatch[b], layers[layerNum - 1], ySplitGPU + yPosBatch[b], mBatch[b]);
	}

	cudaFree(xTransGPU);
	cudaFree(yTransGPU);
}

// Wrapper for cublasSgemm to make life easier C(m,n) = A(m,k) * B(k,n)
void cublasNN::matMatMultiplyGPU(const float *A, const float *B, float *C, const int a, const int b, const int c,
                                 cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N,
                                 int lda = -1, int ldb = -1, int ldc = -1, float alpha = 1, float beta = 0)
{
	//int lda = m, int ldb = k, int ldc = m
	(lda < 0 && (lda = a));
	(ldb < 0 && (ldb = c));
	(ldc < 0 && (ldc = a));

	// Do the actual multiplication
	cublasSgemm(handle, transa, transb, a, b, c, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

double cublasNN::trainFuncApproxGradDescent(float rate, int batchNum /*= 1*/)
{
	return trainFuncApproxMomentum(0, rate, batchNum);
}

double cublasNN::trainFuncApproxMomentum(float momentum, float rate, int batchNum /*= 1*/)
{
	auto start = chrono::steady_clock::now();

	//momentum *= -1; // possible error
	float* velocity;

	cudaMalloc((void**)&velocity, totalThetaSize * sizeof(float));
	cublasSscal(handle, totalThetaSize, &zero, velocity, 1);

	allocVar(batchNum, false);

	splitData(batchNum);

	for(int i = 0; i < iters; i++)
	{
		for(int b = 0; b < batchNum; b++)
		{
			float alpha = -rate / mBatch[b];
			float J = gradFuncApprox(b);
			if((i + 1) % display == 0 && b == 0)
				cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;
			for(int j = 0; j < layerNum - 1; j++)
			{
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &momentum, (velocity + thetaPos[j]),
				            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (velocity + thetaPos[j]),
				            (layers[j] + 1));
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (layers[j] + 1), layers[j + 1], &one, (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1), &one, (velocity + thetaPos[j]), (layers[j] + 1), (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1));
			}
		}
	}

	releaseVar(false);

	calcFinalCost(false);

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

double cublasNN::trainClassifyGradDescent(float rate, int batchNum /*= 1*/)
{
	return trainFuncApproxMomentum(0, rate, batchNum);
}

double cublasNN::trainClassifyMomentum(float momentum, float rate, int batchNum /*= 1*/)
{
	auto start = chrono::steady_clock::now();

	float* velocity;

	cudaMalloc((void**)&velocity, totalThetaSize * sizeof(float));
	cublasSscal(handle, totalThetaSize, &zero, velocity, 1);

	allocVar(batchNum, false);

	splitData(batchNum);

	for(int i = 0; i < iters; i++)
	{
		for(int b = 0; b < batchNum; b++)
		{
			float alpha = -rate / mBatch[b];
			float J = gradClassify(b);
			if((i + 1) % display == 0 && b == 0)
				cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;
			for(int j = 0; j < layerNum - 1; j++)
			{
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &momentum, (velocity + thetaPos[j]),
				            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (velocity + thetaPos[j]),
				            (layers[j] + 1));
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (layers[j] + 1), layers[j + 1], &one, (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1), &one, (velocity + thetaPos[j]), (layers[j] + 1), (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1));
			}
		}
	}

	releaseVar(false);

	calcFinalCost(true);

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

double cublasNN::trainClassifyQuasiNewtonMomentum(float momentum, float rate, int batchNum /*= 1*/)
{
	auto start = chrono::steady_clock::now();

	allocVar(batchNum, true);

	splitData(batchNum);

	float* velocity;

	cudaMalloc((void**)&velocity, thetaSize[layerNum - 2] * sizeof(float));
	cublasSscal(handle, thetaSize[layerNum - 2], &zero, velocity, 1);

	for(int i = 0; i < iters; i++)
	{
		for(int b = 0; b < batchNum; b++)
		{
			const float mReci = 1 / (float)mBatch[b];
			float J = gradClassify(b);
			hessClassify(b);
			//cout << mBatch[b] << endl << (1 / (float)mBatch[b]) << endl << mReci << endl; // mReci shows up as 0;
			cublasSscal(handle, totalThetaSize, &mReci, DeltaBaseGPU, 1);
			/*for(int j = 0; j < layerNum - 1; j++)
			{
				float* temp;
				temp = (float*)malloc(DeltaSize[j] * sizeof(float));
				cudaMemcpy(temp, (DeltaBaseGPU + DeltaPos[j]), DeltaSize[j] * sizeof(float), cudaMemcpyDeviceToHost);
				cout << j << '\t' << DeltaPos[j] << endl;
				for(int k = 0; k < 5; k++) {
					for(int l = 0; l < 5; l++)
						cout << temp[IDX2C(k, l, layers[j + 1])] << '\t';
					cout << endl;
				}
				cout << endl;
				free(temp);
			}*/
			absVecGPU(Delta2BaseGPU, Delta2BaseGPU, totalThetaSize);
			//scaVecAddGPU(Delta2BaseGPU, mu, Delta2BaseGPU, totalThetaSize);
			vecVecElementDivideGPU(DeltaBaseGPU, Delta2BaseGPU, DeltaBaseGPU, totalThetaSize - DeltaSize[layerNum - 2]);
			//vecVecSubtractGPU(thetaBaseGPU, step, thetaBaseGPU, totalThetaSize);
			if((i + 1) % display == 0 && b == 0)
				cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;
			for(int j = 0; j < layerNum - 2; j++)
			{
				/*cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &momentum, (velocity + thetaPos[j]),
				            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (velocity + thetaPos[j]),
				            (layers[j] + 1));*/
				cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &one, (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1), &negOne, (DeltaBaseGPU + thetaPos[j]), layers[j + 1], (thetaBaseGPU + thetaPos[j]),
				            (layers[j] + 1));
			}
			float alpha = -rate;
			int j = layerNum - 2;
			cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &momentum, (velocity + thetaPos[j]),
			            (layers[j] + 1), &alpha, (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]), (velocity + thetaPos[j]),
			            (layers[j] + 1));
			cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (layers[j] + 1), layers[j + 1], &one, (thetaBaseGPU + thetaPos[j]),
			            (layers[j] + 1), &one, (velocity + thetaPos[j]), (layers[j] + 1), (thetaBaseGPU + thetaPos[j]),
			            (layers[j] + 1));
			/*for(int j = 0; j < layerNum - 1; j++)
			{
				float* temp;
				temp = (float*)malloc(thetaSize[j] * sizeof(float));
				cudaMemcpy(temp, (thetaBaseGPU + thetaPos[j]), thetaSize[j] * sizeof(float), cudaMemcpyDeviceToHost);
				cout << j << '\t' << thetaPos[j] << endl;
				for(int k = 0; k < 5; k++) {
					for(int l = 0; l < 5; l++)
						cout << temp[IDX2C(k, l, layers[j] + 1)] << '\t';
					cout << endl;
				}
				cout << endl;
				free(temp);
			}*/
		}
	}

	/*for(int j = 0; j < layerNum - 2; j++)
	{
		float* temp;
		temp = (float*)malloc(DeltaSize[j] * sizeof(float));
		cudaMemcpy(temp, (DeltaBaseGPU + DeltaPos[j]), DeltaSize[j] * sizeof(float), cudaMemcpyDeviceToHost);
		cout << j << '\t' << DeltaPos[j] << endl;
		for(int k = 0; k < 5; k++) {
			for(int l = 0; l < 5; l++)
				cout << temp[IDX2C(k, l, layers[j + 1])] << '\t';
			cout << endl;
		}
		cout << endl;
		free(temp);
	}*/

	releaseVar(true);

	calcFinalCost(true);

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

float cublasNN::gradFuncApprox(int b /*short for batchNum*/)
{
	float J;

	// Forward Propagate
	matMatMultiplyGPU(xSplitGPU + xPosBatch[b], (thetaBaseGPU + thetaPos[0]), (zBaseGPU + zPos[0]),
	                  mBatch[b], layers[1], (layers[0] + 1));
	sigmoidVecGPU((zBaseGPU + zPos[0]), (aBaseGPU + aPos[0] + mBatch[b]), zSize[0]);

	for(int j = 1; j < layerNum - 2; j++)
	{
		onesVecGPU((aBaseGPU + aPos[j - 1]), mBatch[b]);
		matMatMultiplyGPU((aBaseGPU + aPos[j - 1]), (thetaBaseGPU + thetaPos[j]), (zBaseGPU + zPos[j]),
		                  mBatch[b], layers[j + 1], (layers[j] + 1));
		sigmoidVecGPU((zBaseGPU + zPos[j]), (aBaseGPU + aPos[j] + mBatch[b]), zSize[j]);
	}
	onesVecGPU((aBaseGPU + aPos[layerNum - 3]), mBatch[b]);
	matMatMultiplyGPU((aBaseGPU + aPos[layerNum - 3]), (thetaBaseGPU + thetaPos[layerNum - 2]),
	                  (zBaseGPU + zPos[layerNum - 2]), mBatch[b], layers[layerNum - 1], (layers[layerNum - 2] + 1));
	//sigmoidVecGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);

	// Calculate the last delta
	vecVecSubtractGPU((zBaseGPU + zPos[layerNum - 2]), ySplitGPU + yPosBatch[b], (deltaBaseGPU + deltaPos[layerNum - 2]),
	                  zSize[layerNum - 2]);

	// Calculate cost
	cublasSdot(handle, deltaSize[layerNum - 2], (deltaBaseGPU + deltaPos[layerNum - 2]), 1,
	           (deltaBaseGPU + deltaPos[layerNum - 2]), 1, &J);
	J /= (2 * mBatch[b]);

	// Calculate remaining deltas via backpropagation
	for(int j = layerNum - 3; j >= 0; j--)
	{
		sigmoidGradVecGPU((zBaseGPU + zPos[j]), (sigGradBaseGPU + sigGradPos[j]), zSize[j]);
		matMatMultiplyGPU((deltaBaseGPU + deltaPos[j + 1]), (thetaBaseGPU + thetaPos[j + 1]), (productBaseGPU + productPos[j]),
		                  mBatch[b], (layers[j + 1] + 1), layers[j + 2], CUBLAS_OP_N, CUBLAS_OP_T, mBatch[b], (layers[j + 1] + 1),
		                  mBatch[b]);
		vecVecElementMultiplyGPU((productBaseGPU + productPos[j] + mBatch[b]), (sigGradBaseGPU + sigGradPos[j]),
		                         (deltaBaseGPU + deltaPos[j]), deltaSize[j]);
	}

	// Accumulate deltas to calculate Deltas
	matMatMultiplyGPU((deltaBaseGPU + deltaPos[0]), (xSplitGPU + xPosBatch[b]), (DeltaBaseGPU + DeltaPos[0]), layers[1], (layers[0] + 1),
	                  mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], layers[1]);
	for(int j = 1; j < layerNum - 1; j++)
		matMatMultiplyGPU((deltaBaseGPU + deltaPos[j]), (aBaseGPU + aPos[j - 1]), (DeltaBaseGPU + DeltaPos[j]), (layers[j + 1]),
		                  (layers[j] + 1), mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], (layers[j + 1]));

	return J;
}

float cublasNN::gradClassify(int b /*short for batchNum*/)
{
	float J;

	// Forward Propagate
	matMatMultiplyGPU(xSplitGPU + xPosBatch[b], (thetaBaseGPU + thetaPos[0]), (zBaseGPU + zPos[0]),
	                  mBatch[b], layers[1], (layers[0] + 1));
	sigmoidVecGPU((zBaseGPU + zPos[0]), (aBaseGPU + aPos[0] + mBatch[b]), zSize[0]);

	for(int j = 1; j < layerNum - 2; j++)
	{
		onesVecGPU((aBaseGPU + aPos[j - 1]), mBatch[b]);
		matMatMultiplyGPU((aBaseGPU + aPos[j - 1]), (thetaBaseGPU + thetaPos[j]), (zBaseGPU + zPos[j]),
		                  mBatch[b], layers[j + 1], (layers[j] + 1));
		sigmoidVecGPU((zBaseGPU + zPos[j]), (aBaseGPU + aPos[j] + mBatch[b]), zSize[j]);
	}
	onesVecGPU((aBaseGPU + aPos[layerNum - 3]), mBatch[b]);
	matMatMultiplyGPU((aBaseGPU + aPos[layerNum - 3]), (thetaBaseGPU + thetaPos[layerNum - 2]),
	                  (zBaseGPU + zPos[layerNum - 2]), mBatch[b], layers[layerNum - 1], (layers[layerNum - 2] + 1));
	sigmoidVecGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);

	// Calculate the last delta

	vecVecSubtractGPU((aBaseGPU + aPos[layerNum - 2]), ySplitGPU + yPosBatch[b], (deltaBaseGPU + deltaPos[layerNum - 2]),
	                  aSize[layerNum - 2]);

	// Calculate cost
	negLnMaxCostGPU((aBaseGPU + aPos[layerNum - 2]), ySplitGPU + yPosBatch[b], JAll, aSize[layerNum - 2]);
	cublasSasum(handle, aSize[layerNum - 2], JAll, 1, &J);
	J /= mBatch[b];

	// Calculate remaining deltas via backpropagation
	for(int j = layerNum - 3; j >= 0; j--)
	{
		sigmoidGradVecGPU((zBaseGPU + zPos[j]), (sigGradBaseGPU + sigGradPos[j]), zSize[j]);
		matMatMultiplyGPU((deltaBaseGPU + deltaPos[j + 1]), (thetaBaseGPU + thetaPos[j + 1]), (productBaseGPU + productPos[j]),
		                  mBatch[b], (layers[j + 1] + 1), layers[j + 2], CUBLAS_OP_N, CUBLAS_OP_T, mBatch[b], (layers[j + 1] + 1),
		                  mBatch[b]);
		vecVecElementMultiplyGPU((productBaseGPU + productPos[j] + mBatch[b]), (sigGradBaseGPU + sigGradPos[j]),
		                         (deltaBaseGPU + deltaPos[j]), deltaSize[j]);
	}

	// Accumulate deltas to calculate Deltas
	matMatMultiplyGPU((deltaBaseGPU + deltaPos[0]), (xSplitGPU + xPosBatch[b]), (DeltaBaseGPU + DeltaPos[0]), layers[1], (layers[0] + 1),
	                  mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], layers[1]);
	for(int j = 1; j < layerNum - 1; j++)
	matMatMultiplyGPU((deltaBaseGPU + deltaPos[j]), (aBaseGPU + aPos[j - 1]), (DeltaBaseGPU + DeltaPos[j]), layers[j + 1],
	                  (layers[j] + 1), mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], (layers[j + 1]));

	return J;
}

void cublasNN::hessClassify(int b /*short for batchNum*/)
{
	const float mReci = 1 / (float)mBatch[b];
	//onesVecGPU((delta2BaseGPU + delta2Pos[layerNum - 2]), delta2Size[layerNum - 2]);
	sigmoidGradFromResultVecGPU((aBaseGPU + aPos[layerNum - 2]), (delta2BaseGPU + delta2Pos[layerNum - 2]), delta2Size[layerNum - 2]);
	for(int j = layerNum - 3; j >= 0; j--)
	{
		vecVecElementMultiplyGPU((thetaBaseGPU + thetaPos[j + 1]), (thetaBaseGPU + thetaPos[j + 1]), thetaSq, thetaSize[j + 1]);

		matMatMultiplyGPU((delta2BaseGPU + delta2Pos[j + 1]), thetaSq, productHess, mBatch[b], (layers[j + 1] + 1),
		                  layers[j + 2], CUBLAS_OP_N, CUBLAS_OP_T, mBatch[b], (layers[j + 1] + 1), mBatch[b]);
		vecVecElementMultiplyGPU((sigGradBaseGPU + sigGradPos[j]), (sigGradBaseGPU + sigGradPos[j]), sigGradSq, sigGradSize[j]);
		vecVecElementMultiplyGPU(sigGradSq, (productHess + mBatch[b]), (delta2BaseGPU + delta2Pos[j]), delta2Size[j]);
		sigmoidGrad2VecGPU((zBaseGPU + zPos[j]), sigGrad2, zSize[j]);

		vecVecElementMultiplyGPU(sigGrad2, (productBaseGPU + productPos[j] + mBatch[b]), productHess, delta2Size[j]);

		cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, mBatch[b], layers[j + 1], &one, (delta2BaseGPU + delta2Pos[j]), mBatch[b],
		            &one, productHess, mBatch[b], (delta2BaseGPU + delta2Pos[j]), mBatch[b]);
	}

	vecVecElementMultiplyGPU((xSplitGPU + xPosBatch[b]), (xSplitGPU + xPosBatch[b]), aSq, (mBatch[b] * n));
	matMatMultiplyGPU((delta2BaseGPU + delta2Pos[0]), aSq, (Delta2BaseGPU + Delta2Pos[0]), layers[1], (layers[0] + 1),
	                  mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], layers[1], mReci);
	/*cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, (layers[0] + 1), layers[1], &one, (Delta2BaseGPU + Delta2Pos[0]),
	            layers[1], &zero, (Delta2BaseGPU + Delta2Pos[0]), layers[1], (hessianBaseGPU + thetaPos[0]), (layers[0] + 1));*/
	for(int j = 1; j < layerNum - 1; j++)
	{
		vecVecElementMultiplyGPU((aBaseGPU + aPos[j - 1]), (aBaseGPU + aPos[j - 1]), aSq, aSize[j - 1]);
		matMatMultiplyGPU((delta2BaseGPU + delta2Pos[j]), aSq, (Delta2BaseGPU + Delta2Pos[j]), layers[j + 1],
		                  (layers[j] + 1), mBatch[b], CUBLAS_OP_T, CUBLAS_OP_N, mBatch[b], mBatch[b], layers[j + 1], mReci);
		/*cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, (layers[j] + 1), layers[j + 1], &one, (Delta2BaseGPU + Delta2Pos[j]),
	            layers[j + 1], &zero, (Delta2BaseGPU + Delta2Pos[j]), layers[j + 1], (hessianBaseGPU + thetaPos[j]), (layers[j] + 1));*/
	}
}

float cublasNN::calcFinalCost(bool classify)
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
	cudaMalloc((void**)&deltaBaseGPU, (m * layers[layerNum - 1]) * sizeof(float));

		// Forward Propagate
	matMatMultiplyGPU(xGPU, (thetaBaseGPU + thetaPos[0]), (zBaseGPU + zPos[0]), m, layers[1], (layers[0] + 1));
	sigmoidVecGPU((zBaseGPU + zPos[0]), (aBaseGPU + aPos[0] + m), zSize[0]);
	for(int j = 1; j < layerNum - 2; j++)
	{
		onesVecGPU((aBaseGPU + aPos[j - 1]), m);
		matMatMultiplyGPU((aBaseGPU + aPos[j - 1]), (thetaBaseGPU + thetaPos[j]), (zBaseGPU + zPos[j]),
		                  m, layers[j + 1], (layers[j] + 1));
		sigmoidVecGPU((zBaseGPU + zPos[j]), (aBaseGPU + aPos[j] + m), zSize[j]);
	}
	onesVecGPU((aBaseGPU + aPos[layerNum - 3]), m);
	matMatMultiplyGPU((aBaseGPU + aPos[layerNum - 3]), (thetaBaseGPU + thetaPos[layerNum - 2]),
	                  (zBaseGPU + zPos[layerNum - 2]), m, layers[layerNum - 1], (layers[layerNum - 2] + 1));

	for(int j = layerNum - 2; j < layerNum - 1; j++)
	{
		float* temp;
		temp = (float*)malloc(zSize[j] * sizeof(float));
		cudaMemcpy(temp, (zBaseGPU + zPos[j]), zSize[j] * sizeof(float), cudaMemcpyDeviceToHost);
		cout << j << '\t' << zPos[j] << endl;
		for(int k = 0; k < 5; k++) {
			for(int l = 0; l < 5; l++)
				cout << temp[IDX2C(k, l, m)] << '\t';
			cout << endl;
		}
		cout << endl;
		free(temp);
	}

	// Calculate the last delta
	if(classify == true)
	{
		sigmoidVecGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);
		negLnMaxCostGPU((aBaseGPU + aPos[layerNum - 2]), yGPU, JAll, aSize[layerNum - 2]);
		cublasSasum(handle, aSize[layerNum - 2], JAll, 1, &J);
		J /= m;

		float* temp;
		temp = (float*)malloc(aSize[layerNum - 2] * sizeof(float));
		cudaMemcpy(temp, (aBaseGPU + aPos[layerNum - 2]), aSize[layerNum - 2] * sizeof(float), cudaMemcpyDeviceToHost);
		cout << layerNum - 2 << '\t' << aPos[layerNum - 2] << endl;
		for(int k = 0; k < 5; k++)
		{
			for(int l = 0; l < 5; l++)
				cout << temp[IDX2C(k, l, m)] << '\t';
			cout << endl;
		}
		cout << endl;
		free(temp);
	}
	else
	{
		vecVecSubtractGPU((aBaseGPU + aPos[layerNum - 2]), yGPU, deltaBaseGPU, aSize[layerNum - 2]);
		cublasSdot(handle, deltaSize[layerNum - 2], (deltaBaseGPU + deltaPos[layerNum - 2]), 1,
		           (deltaBaseGPU + deltaPos[layerNum - 2]), 1, &J);
		J /= (2 * m);
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

	cudaFree(deltaBaseGPU);

	return J;
}


void cublasNN::releaseVar(bool hessian)
{
	cudaFree(sigGradBaseGPU);
	cudaFree(productBaseGPU);
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

	if(hessian)
	{
		cudaFree(sigGradSq);
		cudaFree(sigGrad2);
		cudaFree(productHess);
		cudaFree(thetaSq);

		cudaFree(delta2BaseGPU);
		free(delta2Pos);
		free(delta2Size);
		
		cudaFree(Delta2BaseGPU);
		free(Delta2Pos);
		free(Delta2Size);
	}
}