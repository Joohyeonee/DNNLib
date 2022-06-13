#include <iostream>
#include <string>
#include <chrono>
#include "csvparser.h"
#include "NewDNN/DNN.h"

void predictionTest();
void classificationTest();
std::vector<std::vector<double>> readFile(const char* aNameOfDataWithDir);

void main()
{
	auto clockbegin = std::chrono::steady_clock::now();
	//predictionTest();
	classificationTest();
	auto clockend = std::chrono::steady_clock::now();
	std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(clockend - clockbegin).count() << "[ms]" << std::endl;

	getchar();
}

void predictionTest()
{
	//#####################################초기 설정#####################################//
	//data
	std::string dataPath = "C:\\Users\\kk\\Documents\\csv\\조업편차분석.csv";
	const char *NameofData = dataPath.c_str();
	std::vector<std::vector<double>> data = readFile(NameofData);
	//nNodes
	std::vector<int> nodeNum = { 53,20,20,1 };
	//Initialize Neural Network
	NeuralNet NN(data, nodeNum, 16, 100, prediction, standard, adam, relu, linear, 0.001, 0.9, 0.99, 0.9, 0.999);
	//Train Process
	int nTrain = data.size() / NN.nBatch;
	//#####################################초기 설정#####################################//
	
	//Normalization
	normalization(NN);
	//train
	for (int epoch = 0; epoch < NN.nEpoch; epoch++)
	{
		for (int train = 0; train < nTrain; train++)
		{
			setInput(NN, train*NN.nBatch, (train + 1)*NN.nBatch);
			setTarget(NN, train*NN.nBatch, (train + 1)*NN.nBatch);
			feedForward(NN);
			backPropagation(NN);
			updateWeight(NN);
		}
		std::cout << epoch + 1 << "epoch RMSE: " << calcRMSE(NN) << std::endl;
		if (calcRMSE(NN) < 0.0001) break;
	}

	//pridict
	std::string dataPath_predict = "C:\\Users\\kk\\Documents\\csv\\조업편차_test.csv";
	const char *NameofData_predict = dataPath_predict.c_str();
	std::vector<std::vector<double>> data_predict = readFile(NameofData_predict);
	NeuralNet model(data_predict);
	makeModel(model, NN);
	normalization(model);
	setInput(model);
	predict(model);
	denormalization(model);
	//#####################################결과 확인#####################################//
	int dataRowNum = model.data.size();
	int dataColNum = model.data[0].size();
	for (int i = 0; i < dataRowNum; i++)
	{
		for (int j = dataColNum - 1; j < dataColNum; j++)
		{
			std::cout << data_predict[i][j] << "[" << model.data[i][j] << "] ";
		}
		std::cout << std::endl;
	}
	//#####################################결과 확인#####################################//
	std::cout << "Test RMSE: " << calcRMSE(model) << std::endl;
}

void classificationTest()
{
	//#####################################초기 설정#####################################//
	//data
	std::string dataPath = "C:\\Users\\kk\\Documents\\csv\\iris DA_int_train.csv";
	const char *NameofData = dataPath.c_str();
	std::vector<std::vector<double>> data = readFile(NameofData);
	//nNodes
	std::vector<int> nodeNum = { 4,20,20,3 };
	//Initialize Neural Network
	NeuralNet NN(data, nodeNum, 10, 200, classification, standard, gdm, relu, softmax, 0.001, 0.9, 0.99, 0.9, 0.999);
	//Train Process
	int nTrain = data.size() / NN.nBatch;
	//#####################################초기 설정#####################################//
	//Normalization
	normalization(NN);
	//train
	for (int epoch = 0; epoch < NN.nEpoch; epoch++)
	{
		for (int train = 0; train < nTrain; train++)
		{
			setInput(NN, train*NN.nBatch, (train + 1)*NN.nBatch);
			setTarget(NN, train*NN.nBatch, (train + 1)*NN.nBatch);
			feedForward(NN);
			backPropagation(NN);
			updateWeight(NN);
		}
		std::cout << epoch + 1 << "epoch RMSE: " << calcRMSE(NN) << std::endl;
		if (calcRMSE(NN) < 0.0001) break;
	}
 
	//pridict
	std::string dataPath_predict = "C:\\Users\\kk\\Documents\\csv\\iris DA_int_test.csv";
	const char *NameofData_predict = dataPath_predict.c_str();
	std::vector<std::vector<double>> data_predict = readFile(NameofData_predict);
	NeuralNet model(data_predict);
	makeModel(model, NN);
	normalization(model);
	setInput(model);
	predict(model);
	denormalization(model);
	//#####################################결과 확인#####################################//
	int dataRowNum = model.data.size();
	int dataColNum = model.data[0].size();
	for (int i = 0; i < dataRowNum; i++)
	{
		for (int j = dataColNum - 1; j < dataColNum; j++)
		{
			std::cout << data_predict[i][j] << "[" << model.data[i][j] << "] ";
		}
		std::cout << std::endl;
	}
	//#####################################결과 확인#####################################//
	double acc = 0;
	for (int i = 0; i < data_predict.size(); i++)
	{
		if (data_predict[i][dataColNum - 1] == model.data[i][dataColNum - 1]) acc++;
	}

	double totNum = data_predict.size();
	double accuracy = acc / totNum;
	std::cout << "Test Accuracy(New Classification): " << accuracy << std::endl;
}

std::vector<std::vector<double>> readFile(const char* aNameOfDataWithDir) {
	std::vector<std::vector<double>> m_A;
	CsvParser *csvparser = CsvParser_new(aNameOfDataWithDir, ",", 1);
	CsvRow *row;

	const CsvRow *header = CsvParser_getHeader(csvparser);
	if (header == NULL) {
		printf("%s\n", CsvParser_getErrorMessage(csvparser));
	}
	m_A.clear();
	const char **headerFields = CsvParser_getFields(header);
	int numXfield = CsvParser_getNumFields(header);

	int TheCounter = 0;
	while ((row = CsvParser_getRow(csvparser))) {
		const char **rowFields = CsvParser_getFields(row);
		TheCounter++;
		//printf("Counting the number of Rows %d\n", TheCounter++);
		CsvParser_destroy_row(row);
	}

	CsvParser_destroy(csvparser);
	TheCounter = 0;

	const CsvRow *header2 = CsvParser_getHeader(csvparser);
	if (header2 == NULL) {
		printf("%s\n", CsvParser_getErrorMessage(csvparser));
	}

	csvparser = CsvParser_new(aNameOfDataWithDir, ",", 1);
	while ((row = CsvParser_getRow(csvparser))) {
		std::vector<double> temp;
		const char **rowFields = CsvParser_getFields(row);
		for (int ind = 0; ind < numXfield; ind++)
		{
			temp.push_back(strtod(rowFields[ind], NULL));

		}
		TheCounter++;
		m_A.push_back(temp);
		CsvParser_destroy_row(row);
	}

	CsvParser_destroy(csvparser);
	int ms = m_A.size();
	return m_A;
}