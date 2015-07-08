#include <stdio.h>
#include <iostream>

#include "cublasNN.h"

int main(int argc, char **argv)
{
	cout << "Neural Network Gradient Descent Tests" << endl;
	cublasNN nn;
	cout << "Read CSV" << endl;
	float csvTime = 0;
	float csvTime2 = 0;
	vector<vector<float>> xVec = nn.readCSV("trainP7_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVec = nn.readCSV("trainYP2_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecValidate = nn.readCSV("validateP7_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVecValidate = nn.readCSV("validateY1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecPredict = nn.readCSV("testPF1_1.csv", false, csvTime2);
	csvTime += csvTime2;
	cout << csvTime << " s" << endl;

	for (size_t i = 0; i < xVec.size(); i++)
		xVec[i].erase(xVec[i].begin() + 1);
	for (size_t i = 0; i < xVecValidate.size(); i++)
		xVecValidate[i].erase(xVecValidate[i].begin() + 1);
	for (size_t i = 0; i < xVecPredict.size(); i++)
		xVecPredict[i].erase(xVecPredict[i].begin() + 1);

	nn.setData(xVec, yVec);
	nn.setValidateData(xVecValidate, yVecValidate);
	nn.setPredictData(xVecPredict);
	nn.normaliseData();
	nn.normaliseValidateData();
	nn.normalisePredictData();
	/*nn.setAlpha(0.00025);
	nn.setIters(1000000);
	nn.setClassify(false);
	nn.setThreads(15); //Optimum for stock clock AMD FX-6100
	vector<int> layers = {10, 40, 160, 1};
	nn.setLayers(layers);
	float concurrentTime = nn.trainConcurrent();
	cout << "Concurrent Training " << concurrentTime << " s" << endl;
	nn.validate();
	cout << "Write CSV" << endl;
	cout << nn.predict("result1.csv") << " s" << endl;
	cout << "Finished, press enter to end" << endl;*/
	getchar();
}