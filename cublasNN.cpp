#include "cublasNN.h"

#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>

cublasNN::cublasNN()
{
}

cublasNN::~cublasNN()
{
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
	float* mat;
	size_t a = data.size();
	size_t b = data[0].size();

	mat = (float*)malloc(a * b * sizeof(float));
	for(size_t i = 0; i < a; i++)
		for(size_t j = 0; j < b; j++)
			mat[IDX2C(j, i, b)] = data[i][j];

	return mat;
}

void cublasNN::setData(vector<vector<float>> xVec, vector<vector<float>> yVec)
{
	m = xVec.size();
	n = xVec[0].size();
	/*for(size_t i = 0; i < 10; i++)
	{
		for(size_t j = 0; j < n; j++)
			cout << xVec[i][j] << '\t';
		cout << endl;
	}*/
	x = vector2dToMat(xVec);
	/*for(size_t i = 0; i < 10; i++)
	{
		for(size_t j = 0; j < n; j++)
			cout << x[IDX2C(j, i, n)] << '\t';
		cout << endl;
	}*/
	y = vector2dToMat(yVec);
}

void cublasNN::setValidateData(vector<vector<float>> xVec, vector<vector<float>> yVec)
{
	mValidate = xVec.size();
	nValidate = xVec[0].size();
	xValidate = vector2dToMat(xVec);
	yValidate = vector2dToMat(yVec);
}

void cublasNN::setPredictData(vector<vector<float>> xVec)
{
	mPredict = xVec.size();
	nPredict = xVec[0].size();
	xPredict = vector2dToMat(xVec);
}

void cublasNN::normalise(float* data, size_t a, size_t b)
{
	float* Mean;
	float* Stddev;
	Mean = mean(data, a, b);
	Stddev = stddev(data, Mean, a, b);

	for(size_t i = 0; i < a; i++)
	{
		for(size_t j = 0; j < b; j++)
		{
			data[IDX2C(j, i, b)] = data[IDX2C(j, i, b)] - Mean[j];
			data[IDX2C(j, i, b)] = data[IDX2C(j, i, b)] / Stddev[j];
		}
	}
	free(Mean);
	free(Stddev);

	for(size_t i = 0; i < 5; i++)
	{
		for(size_t j = 0; j < 5; j++)
			cout << x[IDX2C(j, i, n)] << '\t';
		cout << endl;
	}
	cout << endl;
}

float* cublasNN::mean(float* data, size_t a, size_t b)
{
	float* result;

	result = (float*)malloc(b * sizeof(float));

	for(size_t i = 0; i < b; i++)
		result[i] = 0;

	for(size_t i = 0; i < a; i++)
		for(size_t j = 0; j < b; j++)
			result[j] += data[IDX2C(j, i, b)];

	for(size_t i = 0; i < b; i++)
		result[i] /= a;

	return result;
}

float* cublasNN::stddev(float* data, float* mean, size_t a, size_t b)
{
	float* result;

	result = (float*)malloc(b * sizeof(float));

	for(size_t i = 0; i < b; i++)
		result[i] = 0;

	for(size_t i = 0; i < a; i++)
	{
		for(size_t j = 0; j < b; j++)
		{
			float diff = data[IDX2C(j, i, b)] - mean[j];
			result[j] += diff * diff;
		}
	}

	for(size_t i = 0; i < b; i++)
	{
		result[i] /= (a - 1);
		result[i] = sqrt(result[i]);
	}

	return result;
}


/*Mat cublasNN::randInitialiseWeights(int in, int out)
{
	float epsilon = sqrt(6) / sqrt(1 + in + out);

	Mat weights(in + 1, out, CV_64F);
	cv::theRNG().state = getTickCount();
	cv::randu(weights, 0.0, 1.0);

	weights = abs(weights * 2 * epsilon - epsilon);

	return weights;
}*/