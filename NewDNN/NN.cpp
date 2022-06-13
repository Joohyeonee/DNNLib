#include "NN.h"

void NeuralNet::initLayer()
{
	//Init Input Node, Output Node
	int inputNode = 0;
	if (data[0].size() > 0) inputNode = nNodes[0] - 1;
	input = new double*[nBatch];
	output = new double *[nBatch];
	act_output = new double *[nBatch];
	target = new double *[nBatch];
	for (int i = 0; i < nBatch; i++)
	{
		input[i] = new double[inputNode]();
		output[i] = new double[nNodes[nNodes.size() - 1]]();
		act_output[i] = new double[nNodes[nNodes.size() - 1]]();
		target[i] = new double[nNodes[nNodes.size() - 1]]();
	}
	//Init Hidden Nodes
	hidden = new double**[nHidden];
	act_hidden = new double**[nHidden];
	for (int i = 0; i < nHidden; i++)
	{
		hidden[i] = new double *[nBatch];
		act_hidden[i] = new double *[nBatch];
		for (int j = 0; j < nBatch; j++)
		{
			hidden[i][j] = new double[nNodes[i + 1]]();
			act_hidden[i][j] = new double[nNodes[i + 1]]();
		}
	}
	//init Error
	error = new double**[nHidden + 1];
	error_in = new double**[nHidden + 1];
	for (int i = 0; i < nHidden + 1; i++)
	{
		error[i] = new double *[nBatch];
		error_in[i] = new double *[nBatch];
		for (int j = 0; j < nBatch; j++)
		{
			error[i][j] = new double[nNodes[i + 1]]();
			error_in[i][j] = new double[nNodes[i + 1]]();
		}
	}
}

void NeuralNet::initWeight()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-0.5, 0.5);

	weight = new double**[nHidden + 1];
	dweight = new double**[nHidden + 1];
	opt_weight = new double**[nHidden + 1];
	if(optimizer == adam)opt_weight_adam = new double**[nHidden + 1];
	opt_bias = new double**[nHidden + 1];
	if (optimizer == adam)opt_bias_adam = new double**[nHidden + 1];
	bias = new double*[nHidden + 1];
	dbias = new double**[nHidden + 1];
	for (int i = 0; i < nHidden + 1; i++)
	{
		weight[i] = new double*[nNodes[i]];
		opt_weight[i] = new double*[nNodes[i]];
		if (optimizer == adam)opt_weight_adam[i] = new double*[nNodes[i]];
		dweight[i] = new double*[nNodes[i]];
		opt_bias[i] = new double*[nBatch];
		if (optimizer == adam)opt_bias_adam[i] = new double*[nBatch];
		dbias[i] = new double*[nBatch];
		for (int j = 0; j < nNodes[i]; j++)
		{
			weight[i][j] = new double[nNodes[i + 1]]();
			opt_weight[i][j] = new double[nNodes[i + 1]]();
			if (optimizer == adam)opt_weight_adam[i][j] = new double[nNodes[i + 1]]();
			dweight[i][j] = new double[nNodes[i + 1]]();
			for (int k = 0; k < nNodes[i + 1]; k++)
			{
				weight[i][j][k] = dis(gen);
			}
		}
		bias[i] = new double[nNodes[i + 1]]();
		for (int k = 0; k < nNodes[i + 1]; k++)
		{
			bias[i][k] = 0.01;
		}
		for (int j = 0; j < nBatch; j++)
		{
			opt_bias[i][j] = new double[nNodes[i + 1]]();
			if (optimizer == adam)opt_bias_adam[i][j] = new double[nNodes[i + 1]]();
			dbias[i][j] = new double[nNodes[i + 1]]();
		}
	}
}

