#pragma once
#include <vector>
#include <random>

__declspec(dllexport) enum OPT_SCALE
{
	minmax = 1,
	maxabs = 2,
	standard = 3
};
__declspec(dllexport) enum OPT_OPTIMIZER
{
	gdm = 1,
	rmsprop = 2,
	adagrad = 3,
	adam = 4
};
__declspec(dllexport) enum OPT_ACTFN
{
	linear = 1,
	sigmoid = 2,
	relu = 3,
	softmax = 4
};
__declspec(dllexport) enum OPT_METHOD
{
	prediction = 1,
	classification = 2
};

__declspec(dllexport) class NeuralNet
{
public:
	std::vector<std::vector<double>> data;
	std::vector<int> nNodes;
	std::vector<double> max;
	std::vector<double> min;
	std::vector<double> mean;
	std::vector<double> std;

	double **input;
	double ***hidden;
	double ***act_hidden;
	double **output;
	double **act_output;
	double **target;
	double ***weight;
	double **bias;
	double ***opt_weight;
	double ***opt_bias;
	double ***opt_weight_adam;
	double ***opt_bias_adam;
	double ***dweight;
	double ***dbias;
	double ***error_in;
	double ***error;

	int nHidden;
	int nBatch;
	int nEpoch;
	OPT_METHOD learn_method;
	OPT_SCALE scaler;
	OPT_OPTIMIZER optimizer;
	OPT_ACTFN actf_hidden;
	OPT_ACTFN actf_output;
	double learningRate;
	double momentum;
	double gamma;
	double beta1;
	double beta2;

public:
	NeuralNet(std::vector<std::vector<double>> data) :data(data) {}
	NeuralNet(std::vector<std::vector<double>> data, std::vector<int> nNodes, int nBatch, int nEpoch, OPT_METHOD learn_method, OPT_SCALE scaler, OPT_OPTIMIZER optimizer, OPT_ACTFN actf_hidden, OPT_ACTFN actf_output, double learningRate, double momentum, double gamma, double beta1, double beta2)
		:data(data), nNodes(nNodes), nBatch(nBatch), nEpoch(nEpoch), learn_method(learn_method), scaler(scaler), optimizer(optimizer), actf_hidden(actf_hidden), actf_output(actf_output), learningRate(learningRate), momentum(momentum), gamma(gamma), beta1(beta1), beta2(beta2)
	{
		nHidden = nNodes.size() - 2;
		max.resize(data[0].size(), 0);
		min.resize(data[0].size(), 0);
		mean.resize(data[0].size(), 0);
		std.resize(data[0].size(), 0);
		initLayer();
		initWeight();
	}
	__declspec(dllexport) void initLayer();
	__declspec(dllexport) void initWeight();
};