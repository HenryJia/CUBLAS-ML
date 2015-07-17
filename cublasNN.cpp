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
	/*for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < n; j++)
			cout << xVec[i][j] << '\t';
		cout << endl;
	}*/
	if(xOld)
	{
		free(x);
		free(y);
	}
	x = vector2dToMat(xVec);
	/*for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < n; j++)
			cout << x[IDX2C(j, i, n)] << '\t';
		cout << endl;
	}*/
	y = vector2dToMat(yVec);
	xOld = true;
	/*For loading into GPU
	 * if(xGPUOld)
	{
		cudaFree(xGPU);
		cudaFree(yGPU);
	}
	cudaStat = cudaMalloc ((void**)&xGPU, m * n * sizeof(*x));
	if (cudaStat != cudaSuccess)
	{
		cout << "device memory allocation failed" << endl;
		return;
	}
	 * Unfinished
	 **/
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

	/*for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 5; j++)
			cout << x[IDX2C(i, j, a)] << '\t';
		cout << endl;
	}
	cout << endl;*/
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

	/*for(int i = 0; i < (layerNum - 1); i++)
	{
		cout << thetaSize[i] << endl;
		cout << thetaPos[i] << endl << endl;
	}*/

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
		/*float* temp = (float *)malloc(thetaSize[i] * sizeof(float)); //Debug Code
		cudaMemcpy(temp, theta, thetaSize[i] * sizeof(float), cudaMemcpyDeviceToHost);
		for(int j = (in - 5); j < in; j++)
		{
			if(i != 2)
				for(int k = (out - 5); k < out; k++)
					cout << temp[IDX2C(j, k, in)] << '\t';
			else
				cout << temp[IDX2C(j, 0, in)] << '\t';
			cout << endl;
		}
		cout << endl;
		free(temp);*/
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
	/*for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 5; j++)
			cout << result[IDX2C(i, j, a)] << '\t';
		cout << endl;
	}
	cout << endl;
	for(int j = 0; j < 5; j++)
	{
		for(int k = 0; k < 5; k++)
			cout << data[IDX2C(j, k, a)] << '\t';
		cout << endl;
	}
	cout << endl;*/
	return result;
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
	/*float* temp = (float *)malloc(a * b * sizeof(float)); //Debug Code
	cudaMemcpy(temp, dataGPU, a * b * sizeof(float), cudaMemcpyDeviceToHost);
	for(int j = 0; j < 5; j++)
	{
		for(int k = 0; k < 5; k++)
			cout << temp[IDX2C(j, k, a)] << '\t';
		cout << endl;
	}
	cout << endl;
	free(temp);*/
	return dataGPU;
}

void cublasNN::allocVarGPU()
{
	totalzSize = 0;
	zPos = (int*)malloc((layerNum - 1) * sizeof(*zPos));
	zSize = (int*)malloc((layerNum - 1) * sizeof(*zSize));
	totaldeltaSize = 0;
	deltaPos = (int*)malloc((layerNum - 1) * sizeof(*deltaPos));
	deltaSize = (int*)malloc((layerNum - 1) * sizeof(*deltaSize));

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
	/*for(int i = 0; i < (layerNum - 1); i++)
	{
		cout << zSize[i] << endl;
		cout << zPos[i] << endl << endl;
	}
	cout << totalzSize << endl;*/

	totalaSize = 0;
	aPos = (int*)malloc((layerNum - 1) * sizeof(*aPos));
	aSize = (int*)malloc((layerNum - 1) * sizeof(*aSize));

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
	/*for(int i = 0; i < (layerNum - 1); i++)
	{
		cout << aSize[i] << endl;
		cout << aPos[i] << endl << endl;
	}
	cout << totalaSize << endl;*/

	totalDeltaSize = 0;
	totalthetaGradSize = 0;
	DeltaPos = (int*)malloc((layerNum - 1) * sizeof(*DeltaPos));
	DeltaSize = (int*)malloc((layerNum - 1) * sizeof(*DeltaSize));
	thetaGradPos = (int*)malloc((layerNum - 1) * sizeof(*thetaGradPos));
	thetaGradSize = (int*)malloc((layerNum - 1) * sizeof(*thetaGradSize));

	if(!DeltaPos || !thetaGradPos || !DeltaSize || !thetaGradSize)
	{
		cout << "Malloc Failed" << endl;
		return;
	}

	DeltaPos[0] = 0;
	thetaGradPos[0] = 0;
	for(int i = 0; i < (layerNum - 1); i++)
	{
		DeltaSize[i] =  thetaSize[i];
		DeltaPos[i] =  thetaPos[i];
		thetaGradSize[i] =  thetaSize[i];
		thetaGradPos[i] =  thetaPos[i];
	}
	totalthetaGradSize = totalThetaSize;
	totalDeltaSize = totalThetaSize;
	/*for(int i = 0; i < (layerNum - 1); i++)
	{
		cout << DeltaSize[i] << endl;
		cout << DeltaPos[i] << endl << endl;
	}
	cout << totalDeltaSize << endl;*/

	cudaMalloc((void**)&zBaseGPU, totalzSize * sizeof(float));
	cudaMalloc((void**)&aBaseGPU, totalaSize * sizeof(float));
	cudaMalloc((void**)&deltaBaseGPU, totaldeltaSize * sizeof(float));
	cudaMalloc((void**)&DeltaBaseGPU, totalDeltaSize * sizeof(float));
	cudaMalloc((void**)&thetaGradBaseGPU, totalthetaGradSize * sizeof(float));
}

// C(m,n) = A(m,k) * B(k,n)
void cublasNN::matMatMultiplyGPU(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
	int lda=m,ldb=k,ldc=m;
	const float alpha = 1, beta = 0;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

double cublasNN::trainFuncApprox()
{
	auto start = chrono::steady_clock::now();
	allocVarGPU();
	float J;

	for(int i = 0; i < iters; i++)
	{
		matMatMultiplyGPU(xGPU, (thetaBaseGPU + thetaPos[0]), (zBaseGPU + zPos[0]), m, layers[1], (layers[0] + 1));
		sigmoidVecGPU((zBaseGPU + zPos[0]), (aBaseGPU + aPos[0] + m), zSize[0]);
		for(int j = 1; j < layerNum - 2; j++) {
			addBiasMatGPU((aBaseGPU + aPos[j - 1]), m);
			matMatMultiplyGPU((aBaseGPU + aPos[j - 1]), (thetaBaseGPU + thetaPos[j]), (zBaseGPU + zPos[j]),
			m, layers[j + 1], (layers[j] + 1));
			sigmoidVecGPU((zBaseGPU + zPos[j]), (aBaseGPU + aPos[j] + m), zSize[j]);
		}
		addBiasMatGPU((aBaseGPU + aPos[layerNum - 3]), m);
		matMatMultiplyGPU((aBaseGPU + aPos[layerNum - 3]), (thetaBaseGPU + thetaPos[layerNum - 2]),
		(zBaseGPU + zPos[layerNum - 2]), m, layers[layerNum - 1], (layers[layerNum - 2] + 1));
		//sigmoidVecGPU((zBaseGPU + zPos[layerNum - 2]), (aBaseGPU + aPos[layerNum - 2]), zSize[layerNum - 2]);

		//delta[base + layerNum - 2] = a[base + layerNum - 2] - datay;
		vecVecSubtractGPU((zBaseGPU + zPos[layerNum - 2]), yGPU, (deltaBaseGPU + deltaPos[layerNum - 2]), zSize[layerNum - 2]);
	
		cublasSdot(handle, deltaSize[layerNum - 2], (deltaBaseGPU + deltaPos[layerNum - 2]), 1,
		(deltaBaseGPU + deltaPos[layerNum - 2]), 1, &J);
		J /= (2 * m);

		cout << "Iteration: " << i << "\t" << "Cost: " << J << endl;
	
	}
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

	cudaFree(thetaGradBaseGPU);
	free(thetaGradPos);
	free(thetaGradSize);
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}