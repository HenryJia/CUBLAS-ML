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
	vector<vector<float>> xVec = nn.readCSV("../bikeshare/trainP7_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVec = nn.readCSV("../bikeshare/trainYP2_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecValidate = nn.readCSV("../bikeshare/validateP7_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVecValidate = nn.readCSV("../bikeshare/validateY1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecPredict = nn.readCSV("../bikeshare/testPF1_1.csv", false, csvTime2);
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
	nn.setIters(30000); //30000
	nn.setDisplay(10000); //10000
	int layers[4] = {10, 40, 160, 1}; //The bias unit is auto added by the class.
	nn.setLayers(layers, 4); //This will random initialise the weights
	nn.addBiasData();
	nn.addBiasDataValidate();
	nn.addBiasDataPredict();
	nn.copyDataGPU();
	float concurrentTime = nn.trainFuncApproxGradDescent(0.0025, 2);
	cout << "GPU Training " << concurrentTime << " s" << endl;
	/*nn.validate();
	cout << "Write CSV" << endl;
	cout << nn.predict("result1.csv") << " s" << endl;
	cout << "Finished, press enter to end" << endl;*/
	getchar();
}
